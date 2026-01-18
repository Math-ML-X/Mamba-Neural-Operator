# set gpu device
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
from functools import partial
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from tensorboardX import SummaryWriter

from utils import load_checkpoint, save_checkpoint, ensure_dir
import torchvision
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import datetime
import logging
import shutil
from glob import glob
from typing import Union
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, TensorDataset
from util_metrics import eval_2d

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def setup_ddp(local_rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_model(opt):

    if opt.use_mamba:
        from nn_module.encoder_module_mamba import SpatialEncoder2D
        from nn_module.decoder_module_mamba import PointWiseDecoder2DSimple
    elif opt.use_st:
        from nn_module.encoder_module_st import SpatialEncoder2D
        from nn_module.decoder_module_st import PointWiseDecoder2DSimple
    else:
        from nn_module.encoder_module import SpatialEncoder2D
        from nn_module.decoder_module import PointWiseDecoder2DSimple

    encoder = SpatialEncoder2D(
        3,   # a + xy coordinates
        96,
        256,
        4,
        6,
        res=res,
        use_ln=True,
    )

    decoder = PointWiseDecoder2DSimple(
        256,
        1,
        scale=0.5,
        res=res
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor, h, resolution):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2*h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2*h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


def make_image_grid(a: torch.Tensor, u_pred: torch.Tensor, u_gt: torch.Tensor, out_path,
                    nrow=3):
    b, h, w, c = u_pred.shape   # c = 1

    a = a.detach().cpu().squeeze(-1).numpy()
    u_pred = u_pred.detach().cpu().squeeze(-1).numpy()
    u_gt = u_gt.detach().cpu().squeeze(-1).numpy()

    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrow, nrow),  # creates 8x8 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(nrow*nrow)):
        # Iterating over the grid returns the Axes.
        if im_no % 3 == 0:
            ax.imshow(a[im_no//3])#, cmap='coolwarm')
        elif im_no % 3 == 1:
            ax.imshow(u_pred[im_no//3])#, cmap='coolwarm')
        elif im_no % 3 == 2:
            ax.imshow(u_gt[im_no//3])#, cmap='coolwarm')

        ax.axis('equal')
        #ax.axis('off')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def get_arguments(parser):
    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Specifies learning rate for tuning. (default: 1e-6)'
    )
    parser.add_argument(
        '--resume_training', action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--path_to_resume', type=str,
        default='none', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--eval_mode', action='store_true',
        help='Just load pretrained checkpoint and evaluate'
    )
    parser.add_argument(
        '--is_vis', action='store_true', help='if output samples'
    )
    parser.add_argument(
        '--iters', type=int, default=5000, help='Number of training iterations. (default: 100k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=1000, help='Save model checkpoints every x iterations. (default: 5k)'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=8, help='Size of each batch (default: 16)'
    )
    parser.add_argument(
        '--train_dataset_path', type=str, required=True, help='Path to dataset.'
    )
    parser.add_argument(
        '--test_dataset_path', type=str, required=True, help='Path to dataset.'
    )

    parser.add_argument(
        '--train_seq_num', type=int, default=9000, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_seq_num', type=int, default=1000, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--resolution', type=int, default=64, help='The resolution used for training'
    )
    parser.add_argument(
        '--resolution_query', type=int, default=128, help='The resolution used for training'
    )
    parser.add_argument(
        '--reduce_resolution', type=int, default=2, help='Reduce the resolution'
    )
    parser.add_argument(
        '--reduce_resolution_query', type=int, default=1, help='Reduce the resolution for query'
    )
    parser.add_argument(
        '--diag_query', action='store_true',
    )
    parser.add_argument(
        '--task_name', type=str, default='DEBUG', help='Reduce the resolution'
    )
    parser.add_argument(
        '--use_mamba', action='store_true',
    )
    parser.add_argument(
        '--use_st', action='store_true',
    )
    parser.add_argument(
        '--dist', action='store_true',
    )
    parser.add_argument(
        '--local_rank', type=int, help='Local rank for distributed training', default=None
    )
    parser.add_argument(
        '--local-rank', type=int, help='Local rank for distributed training', default=None
    )
    return parser


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a PDE transformer")
    parser = get_arguments(parser)
    opt = parser.parse_args()
    print('Using following options')
    print(opt)

    local_rank = opt.local_rank if opt.local_rank is not None else opt.local_rank
    if local_rank is None:
        assert opt.dist == False
        local_rank = 0

    if opt.dist:
        # DDP setup
        local_rank = opt.local_rank
        setup_ddp(local_rank)
        world_size = dist.get_world_size()

    # add code for datasets
    if local_rank == 0:
        print('Preparing the data')
    train_data_path = opt.train_dataset_path
    test_data_path = opt.test_dataset_path

    ntrain = opt.train_seq_num
    ntest = opt.test_seq_num
    res = opt.resolution
    res_query = opt.resolution_query
    reduce_res = opt.reduce_resolution
    reduce_res_query = opt.reduce_resolution_query
    task_name = opt.task_name
    use_mamba = opt.use_mamba
    use_st = opt.use_st
    assert not (use_mamba and use_st)
    metric_list = ['nRMSE', 'RMSE', 'Rel_L2_Norm']

    # sub = int((421 - 1) / (res - 1))
    dx = 1./res_query
    _, beta = opt.train_dataset_path.split('beta')
    beta, _ = beta.split('_')

    from h5reader import H5Reader
    # load data
    reader = H5Reader(train_data_path)
    x = reader.read_field('nu')
    y = reader.read_field('tensor')
    data_num, _, res_x, res_y = y.shape

    assert data_num >= ntrain
    x_train = x[1000:1000+ntrain, ::reduce_res, ::reduce_res]
    x_test = x[:ntest, ::reduce_res, ::reduce_res]
    y_train = y[1000:1000+ntrain, :, ::reduce_res, ::reduce_res] # num*1*resolution*resolution
    y_test = y[:ntest, :, ::reduce_res, ::reduce_res]

    if opt.diag_query:
        x_train_query = x[1000:1000+ntrain, (reduce_res_query-1)::reduce_res_query, (reduce_res_query-1)::reduce_res_query]
        x_test_query = x[:ntest, (reduce_res_query-1)::reduce_res_query, (reduce_res_query-1)::reduce_res_query]
        y_train_query = y[1000:1000+ntrain, :, (reduce_res_query-1)::reduce_res_query, (reduce_res_query-1)::reduce_res_query]  # num*1*resolution*resolution
        y_test_query = y[:ntest, :, (reduce_res_query-1)::reduce_res_query, (reduce_res_query-1)::reduce_res_query]
    else:
        x_train_query = x[1000:1000+ntrain, ::reduce_res_query, ::reduce_res_query]
        x_test_query = x[:ntest, ::reduce_res_query, ::reduce_res_query]
        y_train_query = y[1000:1000+ntrain, :, ::reduce_res_query, ::reduce_res_query]  # num*1*resolution*resolution
        y_test_query = y[:ntest, :, ::reduce_res_query, ::reduce_res_query]

    print(f'Data Shape: {x_train.shape}')
    print(f'Data Query Shape: {x_train_query.shape}')

    x_train = torch.as_tensor(x_train.reshape(ntrain, res, res, 1), dtype=torch.float32)
    x_test = torch.as_tensor(x_test.reshape(ntest, res, res, 1), dtype=torch.float32)
    y_train = torch.as_tensor(y_train.reshape(ntrain, res, res, 1), dtype=torch.float32)
    y_test = torch.as_tensor(y_test.reshape(ntest, res, res, 1), dtype=torch.float32)

    x_train_query = torch.as_tensor(x_train_query.reshape(ntrain, res_query, res_query, 1), dtype=torch.float32)
    x_test_query = torch.as_tensor(x_test_query.reshape(ntest, res_query, res_query, 1), dtype=torch.float32)
    y_train_query = torch.as_tensor(y_train_query.reshape(ntrain, res_query, res_query, 1), dtype=torch.float32)
    y_test_query = torch.as_tensor(y_test_query.reshape(ntest, res_query, res_query, 1), dtype=torch.float32)

    gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridx = gridx.reshape(1, res, 1, 1).repeat([1, 1, res, 1])
    gridy = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridy = gridy.reshape(1, 1, res, 1).repeat([1, res, 1, 1])
    grid = torch.cat((gridx, gridy), dim=-1).reshape(1, -1, 2)

    gridx_query = torch.tensor(np.linspace(0, 1, res_query), dtype=torch.float32)
    gridx_query = gridx_query.reshape(1, res_query, 1, 1).repeat([1, 1, res_query, 1])
    gridy_query = torch.tensor(np.linspace(0, 1, res_query), dtype=torch.float32)
    gridy_query = gridy_query.reshape(1, 1, res_query, 1).repeat([1, res_query, 1, 1])
    grid_query = torch.cat((gridx_query, gridy_query), dim=-1).reshape(1, -1, 2)

    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0) + 1e-5
    y_mean = y_train_query.mean(dim=0)
    y_std = y_train_query.std(dim=0) + 1e-5
    y_mean = rearrange(y_mean, 'h w c -> (h w) c')
    y_std = rearrange(y_std, 'h w c -> (h w) c')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        grid = grid.cuda()
        grid_query = grid_query.cuda()
        x_mean, x_std = x_mean.cuda(), x_std.cuda()
        y_mean, y_std = y_mean.cuda(), y_std.cuda()

    if opt.dist:
        train_sampler = DistributedSampler(TensorDataset(x_train, y_train_query))
        train_dataloader = DataLoader(TensorDataset(x_train, y_train_query),
                                      batch_size=opt.batch_size // world_size,
                                      shuffle=False,
                                      sampler=train_sampler)
        test_dataloader = DataLoader(TensorDataset(x_test, y_test_query),
                                     batch_size=opt.batch_size // world_size,
                                     shuffle=False)


    else:
        train_dataloader = DataLoader(TensorDataset(x_train, y_train_query),
                                       batch_size=opt.batch_size,
                                       shuffle=True)
        test_dataloader = DataLoader(TensorDataset(x_test, y_test_query),
                                      batch_size=opt.batch_size,
                                      shuffle=False)

    # instantiate network
    if local_rank == 0:
        print('Building network')
    encoder, decoder = build_model(opt)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if opt.dist:
            encoder = encoder.cuda(local_rank)
            decoder = decoder.cuda(local_rank)
        else:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

    if opt.dist:
        # Wrap the model for DDP
        encoder = DDP(encoder, device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])

    # typically we use tensorboardX to keep track of experiments
    if local_rank == 0:
        writer = SummaryWriter()
    checkpoint_dir = os.path.join(opt.log_dir, 'model_ckpt', task_name)
    if local_rank == 0:
        ensure_dir(checkpoint_dir)
    sample_dir = os.path.join(opt.log_dir, 'samples_darcyflow', task_name)
    if local_rank == 0:
        ensure_dir(sample_dir)

    # save option information to the disk
    if local_rank == 0:
        logger = logging.getLogger("LOG")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (opt.log_dir, 'logging_info_darcy'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('=======Option used=======')
        for arg in vars(opt):
            logger.info(f'{arg}: {getattr(opt, arg)}')

    # save the py script of models
    # script_dir = os.path.join(opt.log_dir, 'script_cache', task_name)
    # if local_rank == 0:
        # ensure_dir(script_dir)
    # shutil.copy('./nn_module/__init__.py', script_dir)
    # shutil.copy('./nn_module/attention_module.py', script_dir)
    # shutil.copy('./nn_module/cnn_module.py', script_dir)
    # shutil.copy('./nn_module/encoder_module.py', script_dir)
    # shutil.copy('./nn_module/decoder_module.py', script_dir)
    # shutil.copy('./nn_module/fourier_neural_operator.py', script_dir)
    #shutil.copy('./nn_module/gnn_module.py', script_dir)
    #shutil.copy('./train_darcy.py', opt.log_dir)

    # load checkpoint if needed/ wanted
    start_n_iter = 0

    # create optimizers
    if opt.path_to_resume != 'none':
        enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=0.0)
        dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=0.0)

        enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                                  div_factor=1e2,
                                  pct_start=0.2,
                                  final_div_factor=1e5,
                                   )
        dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                                   div_factor=1e2,
                                   pct_start=0.2,
                                   final_div_factor=1e5,
                                   )

        print(f'Resuming checkpoint from: {opt.path_to_resume}')
        ckpt = load_checkpoint(opt.path_to_resume)  # custom method for loading last checkpoint
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])

        if opt.resume_training:
            enc_optim.load_state_dict(ckpt['enc_optim'])
            dec_optim.load_state_dict(ckpt['dec_optim'])

            enc_scheduler.load_state_dict(ckpt['enc_sched'])
            dec_scheduler.load_state_dict(ckpt['dec_sched'])

            start_n_iter = ckpt['n_iter']
            if local_rank == 0:
                print("pretrained checkpoint restored, training resumed")
                logger.info("pretrained checkpoint restored, training resumed")

        elif not opt.eval_mode:
            if local_rank == 0:
                print("pretrained checkpoint restored, using tuning mode")
                logger.info("pretrained checkpoint restored, using tuning mode")

        else:
            if local_rank == 0:
                print("pretrained checkpoint restored, using evaluation mode")
                logger.info("pretrained checkpoint restored, using evaluation mode")
    else:
        enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=0.0)
        dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=0.0)

        enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                                  div_factor=1e2,
                                  pct_start=0.2,
                                  final_div_factor=1e5,
                                   )
        dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                                   div_factor=1e2,
                                   pct_start=0.2,
                                   final_div_factor=1e5,
                                   )

        if local_rank == 0:
            print("No pretrained checkpoint, using training from scratch mode")
            logger.info("No pretrained checkpoint, using training from scratch mode")

    # now we start the main loop
    n_iter = start_n_iter

    # for loop going through dataset
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        train_data_iter = iter(train_dataloader)

        while True:
            if not opt.eval_mode:
                encoder.train()
                decoder.train()

                try:
                    data = next(train_data_iter)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    del train_data_iter
                    train_data_iter = iter(train_dataloader)
                    data = next(train_data_iter)

                # data preparation
                x, y = data

                if use_cuda:
                    x, y = x.cuda(), y.cuda()

                # standardize
                x = (x - x_mean) / x_std
                x = rearrange(x, 'b h w c -> b (h w) c')
                y = rearrange(y, 'b h w c -> b (h w) c')

                input_pos = grid.repeat([x.shape[0], 1, 1])
                prop_pos = grid_query.repeat([x.shape[0], 1, 1])

                x = torch.cat((x, input_pos), dim=-1)

                z = encoder.forward(x, input_pos)
                x_out = decoder.forward(z, prop_pos, input_pos)

                x_out = x_out * y_std + y_mean

                pred_loss = nn.MSELoss()(x_out, y)
                loss = pred_loss

                enc_optim.zero_grad()
                dec_optim.zero_grad()

                loss.backward()

                # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.)
                # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.)

                # Unscales gradients and calls
                enc_optim.step()
                dec_optim.step()

                enc_scheduler.step()
                dec_scheduler.step()

                # udpate tensorboardX
                if local_rank == 0:
                    writer.add_scalar('train_loss', loss, n_iter)
                    writer.add_scalar('prediction_loss', pred_loss, n_iter)

                # pbar.set_description(
                #     f'Total loss (1e-4): {loss.item()*1e4:.1f}||'
                #     f'prediction (1e-4): {pred_loss.item()*1e4:.1f}||'
                #     f'Iters: {n_iter}/{opt.iters}')
                print(
                    f'Total loss (1e-4): {loss.item()*1e4:.1f}||'
                    f'prediction (1e-4): {pred_loss.item()*1e4:.1f}||'
                    f'Iters: {n_iter}/{opt.iters}')
                pbar.update(1)
                start_time = time.time()
                n_iter += 1

            if (opt.eval_mode or (n_iter - 1) % opt.ckpt_every == 0 or n_iter >= opt.iters) and (local_rank == 0):
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()
                metric_dict = {}
                for metric_name in metric_list:
                    metric_dict[metric_name] = []
                with torch.no_grad():
                    for j, data in enumerate(tqdm(test_dataloader)):
                        # data preparation
                        x, y = data

                        if use_cuda:
                            x, y = x.cuda(), y.cuda()

                        _x = x.clone()
                        x = (x - x_mean) / x_std
                        x = rearrange(x, 'b h w c -> b (h w) c')
                        y = rearrange(y, 'b h w c -> b (h w) c')

                        input_pos = grid.repeat([x.shape[0], 1, 1])
                        prop_pos = grid_query.repeat([x.shape[0], 1, 1])

                        x = torch.cat((x, input_pos), dim=-1)  # concat coordinates as additional feature

                        z = encoder.forward(x, input_pos)
                        x_out = decoder.forward(z, prop_pos, input_pos)

                        x_out = x_out * y_std + y_mean

                        metrics = eval_2d(x_out, y, metric_list=metric_list)
                        for metric_name in metric_list:
                            metric_dict[metric_name].append(metrics[metric_name])

                        # save for vis
                        if j == 0 and opt.is_vis:
                            x_out_array = x_out.cpu().numpy().reshape(-1, 64, 64, 1)[0]
                            y_array = y.cpu().numpy().reshape(-1, 64, 64, 1)[0]
                            _x = _x.cpu().numpy().reshape(-1, 64, 64, 1)[0]
                            result_path = os.path.join(sample_dir)
                            mkdir(result_path)
                            np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                                     input=_x,
                                     pred=x_out_array,
                                     gt=y_array)

                metrics_ave_dict = {}
                for metric_name in metric_list:
                    metrics_ave_dict[metric_name] = np.mean(metric_dict[metric_name])

                metric_log = ''
                if not opt.eval_mode:
                    metric_log += 'Iter: {}'.format(n_iter)
                for metric_name in metric_list:
                    metric_log += 'Val {}: {}; '.format(metric_name, metrics_ave_dict[metric_name])
                print(metric_log)

                if not opt.eval_mode and local_rank == 0:
                    # save checkpoint if needed
                    ckpt = {
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'n_iter': n_iter,
                        'enc_optim': enc_optim.state_dict(),
                        'dec_optim': dec_optim.state_dict(),
                        'enc_sched': enc_scheduler.state_dict(),
                        'dec_sched': dec_scheduler.state_dict(),
                    }
                    save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'darcy_beta{beta}_latest_model.ckpt'))
                    save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'darcy_beta{beta}_{n_iter}_model.ckpt'))
                    print('Iteration {}: save checkpoint to {}'.format(n_iter, checkpoint_dir))

                    del ckpt
                if opt.eval_mode or (n_iter >= opt.iters):
                    print('Running finished...')
                    if opt.dist:
                        cleanup_ddp()
                    exit()

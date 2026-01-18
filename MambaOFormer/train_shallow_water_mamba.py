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
        from nn_module.encoder_module_mamba import SpatialTemporalEncoder2D
        from nn_module.decoder_module_mamba import PointWiseDecoder2D
    elif opt.use_st:
        from nn_module.encoder_module_st import SpatialTemporalEncoder2D
        from nn_module.decoder_module_st import PointWiseDecoder2D
    else:
        from nn_module.encoder_module import SpatialTemporalEncoder2D
        from nn_module.decoder_module import PointWiseDecoder2D


    encoder = SpatialTemporalEncoder2D(
        opt.in_channels,
        opt.encoder_emb_dim,
        opt.out_seq_emb_dim,
        opt.encoder_heads,
        opt.encoder_depth,
    )

    decoder = PointWiseDecoder2D(
        opt.decoder_emb_dim,
        opt.out_channels,
        opt.out_step,
        opt.propagator_depth,
        scale=opt.fourier_frequency,
        dropout=0.0,
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


# adapted from Galerkin Transformer
def central_diff(x: torch.Tensor):
    # assuming PBC
    # x: (batch, seq_len, n), h is the step size, assuming n = h*w
    # x = rearrange(x, 'b t (h w) -> b t h w', h=64, w=64)
    x = rearrange(x, 'b t (h w) -> b t h w', h=128, w=128)
    h = 1. / 128.
    x = F.pad(x,
              (1, 1, 1, 1), mode='circular')  # [b t h+2 w+2]
    grad_x = (x[..., 1:-1, 2:] - x[..., 1:-1, :-2]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[..., 2:, 1:-1] - x[..., :-2, 1:-1]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y


def make_image_grid(image: torch.Tensor, out_path, nrow=25):
    b, t, h, w = image.shape
    image = image.detach().cpu().numpy()
    image = image.reshape((b * t, h, w))
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b * t // nrow, nrow),  # creates 2x2 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b * t)):
        # Iterating over the grid returns the Axes.
        ax.imshow(image[im_no])
        ax.axis('off')
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

    # general option
    parser.add_argument(
        '--in_seq_len', type=int, default=10, help='Length of input sequence. (default: 10)'
    )
    # model options for encoder

    parser.add_argument(
        '--in_channels', type=int, default=12, help='Channel of input feature. (default: 3)'
    )
    parser.add_argument(
        '--encoder_emb_dim', type=int, default=64, help='Channel of token embedding in encoder. (default: 128)'
    )
    parser.add_argument(
        '--out_seq_emb_dim', type=int, default=64, help='Channel of output feature map. (default: 128)'
    )
    parser.add_argument(
        '--encoder_depth', type=int, default=2, help='Depth of transformer in encoder. (default: 2)'
    )
    parser.add_argument(
        '--encoder_heads', type=int, default=1, help='Heads of transformer in encoder. (default: 4)'
    )

    # model options for decoder
    parser.add_argument(
        '--out_channels', type=int, default=1, help='Channel of output. (default: 1)'
    )
    parser.add_argument(
        '--decoder_emb_dim', type=int, default=128, help='Channel of token embedding in decoder. (default: 128)'
    )
    parser.add_argument(
        '--out_step', type=int, default=1, help='How many steps to propagate forward each call. (default: 10)'
    )
    parser.add_argument(
        '--out_seq_len', type=int, default=91, help='Length of output sequence. (default: 40)'
    )
    parser.add_argument(
        '--propagator_depth', type=int, default=1, help='Depth of mlp in propagator. (default: 2)'
    )
    parser.add_argument(
        '--decoding_depth', type=int, default=2, help='Depth of decoding network in the decoder. (default: 2)'
    )
    parser.add_argument(
        '--fourier_frequency', type=int, default=8, help='Fourier feature frequency. (default: 8)'
    )
    parser.add_argument(
        '--use_grad', action='store_true',
    )
    parser.add_argument(
        '--curriculum_steps', type=int, default=10, help='at initial stage, dont rollout too long'
    )
    parser.add_argument(
        '--curriculum_ratio', type=float, default=0.0, help='how long is the initial stage?'
    )
    parser.add_argument(
        '--aug_ratio', type=float, default=0.0, help='Probability to randomly crop'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=8, help='Size of each batch (default: 16)'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='Path to dataset.'
    )

    parser.add_argument(
        '--train_seq_num', type=int, default=900, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_seq_num', type=int, default=100, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--reduce_resolution', type=int, default=1, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--reduce_resolution_t', type=int, default=1, help='How many sequences in the training dataset.'
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

    task_name = opt.task_name
    use_mamba = opt.use_mamba
    use_st = opt.use_st
    metric_list = ['nRMSE', 'RMSE', 'Rel_L2_Norm']

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
    sample_dir = os.path.join(opt.log_dir, 'samples_sw2d', task_name)
    if local_rank == 0:
        ensure_dir(sample_dir)

    # save option information to the disk
    if local_rank == 0:
        logger = logging.getLogger("LOG")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (opt.log_dir, 'logging_info_diffision_reaction'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('=======Option used=======')
        for arg in vars(opt):
            logger.info(f'{arg}: {getattr(opt, arg)}')

    # save the py script of models
    # script_dir = os.path.join(opt.log_dir, 'script_cache')
    # if local_rank == 0:
        # ensure_dir(script_dir)
    # shutil.copy('../../nn_module/__init__.py', script_dir)
    # shutil.copy('./nn_module/attention_module.py', script_dir)
    # shutil.copy('./nn_module/cnn_module.py', script_dir)
    # shutil.copy('./nn_module/encoder_module.py', script_dir)
    # shutil.copy('./nn_module/decoder_module.py', script_dir)
    # shutil.copy('./nn_module/fourier_neural_operator.py', script_dir)
    # shutil.copy('../../nn_module/gnn_module.py', script_dir)
    # shutil.copy('../../tune_navier_stokes.py', opt.log_dir)

    # load checkpoint if needed/ wanted
    start_n_iter = 0

    # create optimizers
    if opt.path_to_resume != 'none':
        enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=0.0)
        dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=0.0)

        if opt.resume_training:
            enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                                       div_factor=1e4,
                                       final_div_factor=1e4,
                                       )
            dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                                       div_factor=1e4,
                                       final_div_factor=1e4,
                                       )
        else:
            enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                                       div_factor=20,
                                       pct_start=0.05,
                                       final_div_factor=1e3,
                                       )
            dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                                       div_factor=20,
                                       pct_start=0.05,
                                       final_div_factor=1e3,
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
                                   div_factor=1e4,
                                   final_div_factor=1e4,
                                   )
        dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                                   div_factor=1e4,
                                   final_div_factor=1e4,
                                   )

        if local_rank == 0:
            print("No pretrained checkpoint, using training from scratch mode")
            logger.info("No pretrained checkpoint, using training from scratch mode")


    def read_save_data(data_path, reduction_res, reduction_res_t, fname=None):
        data_paths_all = sorted(glob(os.path.join(data_path, '*.npz')))
        data = {}
        i = 0
        for path in data_paths_all:
            raw_data = np.load(path)
            u_sampled = torch.from_numpy(raw_data['data'][()])  # (101, 128, 128, 1)
            u_sampled = u_sampled[:, :, :, :]
            u_sampled = rearrange(u_sampled.clone(), 't x y a -> x y t a')
            f = u_sampled[::reduction_res, ::reduction_res, ::reduction_res_t, :]  # [128, 128, 101, 2]
            f = rearrange(f, 'x y t a -> (x y) t a')  # (128*128, 101, 1)

            p = torch.from_numpy(raw_data['grid_x'][()])[::reduction_res]  # (128)
            v = torch.from_numpy(raw_data['grid_y'][()])[::reduction_res]  # (128)
            t = torch.from_numpy(raw_data['grid_t'][()])  # (101,)

            gx, gy = torch.meshgrid(p, v)
            grid = torch.stack((gy, gx), dim=-1).reshape(-1, 2)  # [12928, 2]

            if 'data' not in data.keys():
                data['data'] = torch.unsqueeze(f, dim=-1)
                data['grid_xy'] = torch.unsqueeze(grid, dim=-1)
                data['grid_t'] = torch.unsqueeze(t, dim=-1)
            else:
                data['data'] = torch.cat([data['data'], torch.unsqueeze(f, dim=-1)], dim=-1)
                data['grid_xy'] = torch.cat([data['grid_xy'], torch.unsqueeze(grid, dim=-1)], dim=-1)
                data['grid_t'] = torch.cat([data['grid_t'], torch.unsqueeze(t, dim=-1)], dim=-1)

            i += 1
            if i % 100 == 0:
                print('loading {}-th data, shape of data {}, shape of grid_xy {}, shape of t {}'.format(
                    i, data['data'].shape, data['grid_xy'].shape, data['grid_t'].shape))
                print('shape of a single data {}'.format(f.shape))
            if i == opt.train_seq_num:
                break

        return data


    # now we start the main loop
    n_iter = start_n_iter

    data_path = opt.dataset_path
    ntrain = opt.train_seq_num
    ntest = opt.test_seq_num
    reduction_res = opt.reduce_resolution
    reduction_res_t = opt.reduce_resolution_t

    data = read_save_data(data_path, reduction_res, reduction_res_t)
    print('data shape {}, grid shape: {}, t shape: {}'.format(data['data'].shape, data['grid_xy'].shape,
                                                              data['grid_t'].shape))
    x_train = data['data'][:, :opt.in_seq_len, :, :ntrain]  # input: a(x), shape: in_seq_len*(x*y)*a*sample_num
    y_train = data['data'][:, opt.in_seq_len:opt.in_seq_len + opt.out_seq_len, :, :ntrain]  # solution: u(x)

    data_test = read_save_data(data_path.replace('train', 'test'), reduction_res, reduction_res_t)
    x_test = data_test['data'][:, :opt.in_seq_len, :, :ntest]
    y_test = data_test['data'][:, opt.in_seq_len:opt.in_seq_len + opt.out_seq_len, :, :ntest]

    t, a = x_train.shape[1], x_train.shape[2]  # t=101, a=1
    x_train = rearrange(x_train, 'n t a b -> b n (t a)')
    x_test = rearrange(x_test, 'n t a b -> b n (t a)')
    y_train = rearrange(y_train, 'n t a b -> b n (t a)')
    y_test_all = rearrange(data_test['data'], 'n t a b -> b n (t a)')
    y_test = rearrange(y_test, 'n t a b -> b n (t a)')
    # del data

    # gaussian normalization
    x_mean = torch.mean(x_train).unsqueeze(0)  # [1, t_in, hw]
    x_std = torch.std(x_train).unsqueeze(0)  # [1, t_in, hw]

    y_mean = torch.mean(y_train).unsqueeze(0)  # [1, t_out, hw]
    y_std = torch.std(y_train).unsqueeze(0)  # [1, t_out, hw]

    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    x_test = (x_test - x_mean) / x_std

    if use_cuda:
        x_mean, x_std, y_mean, y_std = x_mean.cuda(), x_std.cuda(), y_mean.cuda(), y_std.cuda()

    grid_train = rearrange(data['grid_xy'], 'h a b -> b h a')
    grid_test = rearrange(data_test['grid_xy'], 'h a b -> b h a')

    if opt.dist:
        train_sampler = DistributedSampler(TensorDataset(x_train, y_train, grid_train))
        train_dataloader = DataLoader(TensorDataset(x_train, y_train, grid_train),
                                      batch_size=opt.batch_size // world_size,
                                      shuffle=False,
                                      sampler=train_sampler)
        test_dataloader = DataLoader(TensorDataset(x_test, y_test, y_test_all, grid_test),
                                     batch_size=opt.batch_size // world_size,
                                     shuffle=False)
    else:
        train_dataloader = DataLoader(TensorDataset(x_train, y_train, grid_train),
                                      batch_size=opt.batch_size,
                                      shuffle=True)
        test_dataloader = DataLoader(TensorDataset(x_test, y_test, y_test_all, grid_test),
                                     batch_size=opt.batch_size,
                                     shuffle=False)
    del data

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
                in_seq, gt, grid = data

                input_pos = prop_pos = grid

                if use_cuda:
                    in_seq = in_seq.cuda()
                    gt = gt.cuda()

                    input_pos = input_pos.cuda()
                    prop_pos = prop_pos.cuda()

                if np.random.uniform() > (1 - opt.aug_ratio):
                    sampling_ratio = np.random.uniform(0.45, 0.95)
                    input_idx = torch.as_tensor(
                        np.concatenate(
                            [np.random.choice(input_pos.shape[1], int(sampling_ratio * input_pos.shape[1]),
                                              replace=False).reshape(1, -1)
                             for _ in range(in_seq.shape[0])], axis=0)
                    ).view(in_seq.shape[0], -1).cuda()

                    in_seq = index_points(in_seq, input_idx)
                    input_pos = index_points(input_pos, input_idx)

                in_seq = torch.cat((in_seq, input_pos), dim=-1)

                z = encoder.forward(in_seq, input_pos)

                if opt.dist:
                    x_out = decoder.module.rollout(z, prop_pos, opt.out_seq_len, input_pos)
                else:
                    x_out = decoder.rollout(z, prop_pos, opt.out_seq_len, input_pos)

                x_out = rearrange(x_out, 'b t c -> b c t')

                x_out = rearrange(x_out, 'b n (t c) -> b n t c', t=opt.out_seq_len, c=opt.out_channels)
                gt = rearrange(gt, 'b n (t c) -> b n t c', t=opt.out_seq_len, c=opt.out_channels)

                pred_loss = nn.MSELoss()(x_out, gt)
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
                #     f'Iters: {n_iter}/{opt.iters} ||'
                #     f'Total (1e-4): {loss.item() * 1e4:.1f}||'
                #     f'pred (1e-4): {pred_loss.item() * 1e4:.1f}||'
                #     f'lr (1e-3): {enc_scheduler.get_last_lr()[0] * 1e3:.4f}||'
                #     f'Seq len: {gt.shape[1]}||'
                # )
                print(
                    f'Iters: {n_iter}/{opt.iters} ||'
                    f'Total (1e-4): {loss.item() * 1e4:.1f}||'
                    f'pred (1e-4): {pred_loss.item() * 1e4:.1f}||'
                    f'lr (1e-3): {enc_scheduler.get_last_lr()[0] * 1e3:.4f}||'
                    f'Seq len: {gt.shape[1]}||'
                )

                pbar.update(1)
                n_iter += 1

            if (opt.eval_mode or ((n_iter - 1) % opt.ckpt_every == 0 or n_iter >= opt.iters)) and (local_rank == 0):
                logger.info('Testing')
                print('Testing')

                encoder.eval()
                decoder.eval()
                metric_dict = {}
                for metric_name in metric_list:
                    metric_dict[metric_name] = []
                with torch.no_grad():
                    for j, data in enumerate(tqdm(test_dataloader)):
                        # data preparation
                        in_seq, gt, gt_full, grid = data

                        input_pos = prop_pos = grid

                        if use_cuda:
                            in_seq = in_seq.cuda()
                            gt = gt.cuda()
                            gt_full = gt_full.cuda()

                            input_pos = input_pos.cuda()
                            prop_pos = prop_pos.cuda()

                        in_seq = torch.cat((in_seq, input_pos), dim=-1)

                        z = encoder.forward(in_seq, input_pos)
                        if opt.dist:
                            x_out = decoder.module.rollout(z, prop_pos, opt.out_seq_len, input_pos)
                        else:
                            x_out = decoder.rollout(z, prop_pos, opt.out_seq_len, input_pos)

                        x_out = rearrange(x_out, 'b t c -> b c t')

                        x_out = x_out * y_std + y_mean  # denormalize

                        x_out = rearrange(x_out, 'b n (t c) -> b n t c', t=opt.out_seq_len, c=opt.out_channels)
                        gt = rearrange(gt, 'b n (t c) -> b n t c', t=opt.out_seq_len, c=opt.out_channels)
                        gt_full = rearrange(gt_full, 'b n (t c) -> b n t c', t=opt.out_seq_len + opt.in_seq_len,
                                            c=opt.out_channels)

                        x_out_all = gt_full.clone()
                        x_out_all[:, :, opt.in_seq_len:opt.in_seq_len + opt.out_seq_len, :] = x_out

                        metrics = eval_2d(x_out_all, gt_full, metric_list=metric_list)
                        for metric_name in metric_list:
                            metric_dict[metric_name].append(metrics[metric_name])

                        # save for vis
                        if j == 0 and opt.is_vis:
                            x_out_all_array = x_out_all.cpu().numpy().reshape(-1, 128, 128, 101, 1)[0]
                            gt_full_array = gt_full.cpu().numpy().reshape(-1, 128, 128, 101, 1)[0]
                            result_path = os.path.join(sample_dir)
                            mkdir(result_path)
                            np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                                     pred=x_out_all_array,
                                     gt=gt_full_array)

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
                    save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'sw2d_latest_model.ckpt'))
                    save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'sw2d_{n_iter}_model.ckpt'))
                    print('Iteration {}: save checkpoint to {}'.format(n_iter, checkpoint_dir))

                    del ckpt
                if opt.eval_mode or (n_iter >= opt.iters):
                    print('Running finished...')
                    if opt.dist:
                        cleanup_ddp()
                    exit()

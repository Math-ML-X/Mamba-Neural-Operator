#!/usr/bin/env python
#-*- coding:utf-8 _*-
import sys
import os
sys.path.append('../..')
sys.path.append('..')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import time
import pickle
import numpy as np
import torch
import torch.nn as nn


from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from args import get_args
from data_utils import get_dataset_new, get_model, get_loss_func, collate_op, MIODataLoader
from utils import get_seed, get_num_params
from models.optimizer import Adam, AdamW
from einops import rearrange

from utils_pack.util_common import *
from utils_pack.util_metrics import eval_2d

'''
    A general code framework for training neural operator on irregular domains
'''

EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']


def train(model,
          loss_func,
          metric_func,
          metric_list,
          train_loader,
          valid_loader,
          optimizer,
          lr_scheduler,
          in_seq_len=10,
          out_seq_len=91,
          epochs=10,
          writer=None,
          device="cuda",
          patience=10,
          grad_clip=0.999,
          start_epoch: int = 0,
          print_freq: int = 20,
          model_save_path='./data/checkpoints/',
          save_mode='state_dict',  # 'state_dict' or 'entire'
          model_name='model.pt',
          result_name='result.pt',
          ):

    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    result = None
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = np.inf
    best_val_epoch = 0
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__)for s in EPOCH_SCHEDULERS)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        torch.cuda.empty_cache()
        for batch in train_loader:

            loss = train_batch(model=model,
                               loss_func=loss_func,
                               data=batch,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               in_seq_len=in_seq_len,
                               out_seq_len=out_seq_len,
                               device=device,
                               grad_clip=grad_clip
                           )
            loss = np.array(loss)
            loss_epoch.append(loss)
            it += 1
            lr = optimizer.param_groups[0]['lr']
            lr_history.append(lr)
            log = f"epoch: [{epoch+1}/{end_epoch}]"
            if loss.ndim == 0:  # 1 target loss
                _loss_mean = np.mean(loss_epoch)
                log += " loss: {:.6f}".format(_loss_mean)
            else:
                _loss_mean = np.mean(loss_epoch, axis=0)
                for j in range(len(_loss_mean)):
                    log += " | loss {}: {:.6f}".format(j, _loss_mean[j])
            log += " | current lr: {:.3e}".format(lr)

            if it % print_freq==0:
                print(log)

            if writer is not None:
                writer.add_scalar("train_loss", _loss_mean, it)    #### loss 0 seems to be the sum of all loss

        loss_train.append(_loss_mean)
        loss_epoch = []

        val_result = validate_epoch(model=model,
                                    metric_func=metric_func,
                                    metric_list=metric_list,
                                    valid_loader=valid_loader,
                                    in_seq_len=in_seq_len,
                                    out_seq_len=out_seq_len,
                                    y_normalizer=y_normalizer,
                                    device=device)

        val_metric_name = 'nRMSE'
        val_metric = np.mean(val_result[val_metric_name])
        loss_val.append(val_metric)

        if val_metric < best_val_metric:
            best_val_epoch = epoch
            best_val_metric = val_metric

        if lr_scheduler and is_epoch_scheduler:
            if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                lr_scheduler.step(val_metric)
            else:
                lr_scheduler.step()

        log = ''
        for metric_name in metric_list:
            log += '| val metric {} : {:.6f} '.format(metric_name, np.mean(val_result[metric_name]))

        if writer is not None:
            for metric_name in metric_list:
                log += '| val metric {} : {:.6f} '.format(metric_name, np.mean(val_result[metric_name]))

        log += "| best val: {:.6f} at epoch {} | current lr: {:.3e}".format(best_val_metric, best_val_epoch+1, lr)

        desc_ep = ""
        if _loss_mean.ndim == 0:  # 1 target loss
            desc_ep += "| loss: {:.6f}".format(_loss_mean)
        else:
            for j in range(len(_loss_mean)):
                if _loss_mean[j] > 0:
                    desc_ep += "| loss {}: {:.3e}".format(j, _loss_mean[j])

        desc_ep += log
        print(desc_ep)

        result = dict(
            best_val_epoch=best_val_epoch,
            best_val_metric=best_val_metric,
            loss_train=np.asarray(loss_train),
            loss_val=np.asarray(loss_val),
            lr_history=np.asarray(lr_history),
            # best_model=best_model_state_dict,
            optimizer_state=optimizer.state_dict()
        )
        pickle.dump(result, open(os.path.join(model_save_path, result_name),'wb'))
    return result




def train_batch(model,
                loss_func,
                data,
                optimizer,
                lr_scheduler,
                in_seq_len,
                out_seq_len,
                device,
                grad_clip=0.999):
    optimizer.zero_grad()

    g, u_p, g_u = data

    g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

    bs = g.batch_size
    n = g.num_nodes() // bs

    out = model(g, u_p, g_u)

    gt = rearrange(g.ndata['y'], 'bn (t c) 1 -> bn t c', t=in_seq_len + out_seq_len, c=4)
    out = rearrange(out, 'bn (t c) -> bn t c', t=out_seq_len, c=4)

    out_full = gt.clone()
    out_full[:, in_seq_len:in_seq_len + out_seq_len, :] = out
    y_pred = out_full
    y = gt
    y_pred = rearrange(y_pred, '(b n) t c -> b n t c', b=bs)
    y = rearrange(y, '(b n) t c -> b n t c', b=bs)

    loss = loss_func(y_pred, y)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item()


def validate_epoch(model,
                   metric_func,
                   metric_list,
                   valid_loader,
                   in_seq_len,
                   out_seq_len,
                   y_normalizer,
                   device,
                   ):
    model.eval()
    metric_dict = {}
    for metric_name in metric_list:
        metric_dict[metric_name] = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            g, u_p, g_u = data
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

            bs = g.batch_size

            out = model(g, u_p, g_u)

            gt = rearrange(g.ndata['y'], 'bn (t c) 1 -> bn t c', t=in_seq_len + out_seq_len, c=4)
            out = rearrange(out, 'bn (t c) -> bn t c', t=out_seq_len, c=4)

            out_full = gt.clone()
            out_full[:, in_seq_len:in_seq_len + out_seq_len, :] = out
            y_pred = out_full
            y = gt
            y_pred = rearrange(y_pred, '(b n) t c -> b n t c', b=bs)
            y = rearrange(y, '(b n) t c -> b n t c', b=bs)

            y_pred = y_normalizer.transform(y_pred, inverse=True)
            y = y_normalizer.transform(y, inverse=True)

            metrics = metric_func(y_pred, y, metric_list=metric_list)
            for metric_name in metric_list:
                metric_dict[metric_name].append(metrics[metric_name])

    return metric_dict


if __name__ == "__main__":

    args = get_args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    # args.dataset = 'cns2d'

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    timestamp = time.strftime('%Y%m%d%H%M%S')
    save_path = os.path.join('runs', args.comment + '_' + timestamp)
    mkdir(save_path)
    mkdir(os.path.join(save_path, 'checkpoints'))

    train_dataset, test_dataset = get_dataset_new(args)
    # test_dataset = get_dataset(args)

    train_loader = MIODataLoader(train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=False)
    test_loader = MIODataLoader(test_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                drop_last=False)

    args.space_dim = 2
    y_normalizer = train_dataset.y_normalizer.to(device) if train_dataset.y_normalizer is not None else None

    get_seed(args.seed)
    torch.cuda.empty_cache()

    loss_func = nn.MSELoss()
    metric_list = ['RMSE', 'nRMSE', 'Rel_L2_Norm']
    metric_func = eval_2d

    model = get_model(args)
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    path_prefix = args.dataset + '_{}_'.format(args.component) + model.__name__ + args.comment + time.strftime('_%m%d_%H_%M_%S')
    model_path, result_path = path_prefix + '.pt', path_prefix + '.pkl'

    print(f"Saving model and result in {os.path.join(save_path, 'checkpoints', 'model_path')}\n")

    if args.use_tb:
        writer_path = os.path.join(save_path, 'logs_{}'.format(path_prefix))
        log_path = os.path.join(writer_path, 'params.txt')
        writer = SummaryWriter(log_dir=writer_path)
        fp = open(log_path, "w+")
        sys.stdout = fp

    else:
        writer = None
        log_path = None

    print(model)
    # print(config)

    epochs = args.epochs
    lr = args.lr

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay,betas=(0.9,0.999))
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay,betas=(0.9, 0.999))
    else:
        raise NotImplementedError

    if args.lr_method == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=epochs)
    elif args.lr_method == 'step':
        print('Using step learning rate schedule')
        scheduler = StepLR(optimizer, step_size=args.lr_step_size*len(train_loader), gamma=0.7)
    elif args.lr_method == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = LambdaLR(optimizer, lambda steps: min((steps+1)/(args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader)/float(steps + 1), 0.5)))
    else:
        raise NotImplementedError
    time_start = time.time()

    result = train(model=model,
                   loss_func=loss_func,
                   metric_func=metric_func,
                   metric_list=metric_list,
                   train_loader=train_loader,
                   valid_loader=test_loader,
                   optimizer=optimizer,
                   lr_scheduler=scheduler,
                   in_seq_len=args.dataset_config['in_seq'],
                   out_seq_len=args.dataset_config['out_seq'],
                   epochs=epochs,
                   writer=writer,
                   device=device,
                   grad_clip=args.grad_clip,
                   patience=None,
                   model_name=model_path,
                   model_save_path=os.path.join(save_path, 'checkpoints'),
                   result_name=result_path,
                   )

    print('Training takes {} seconds.'.format(time.time() - time_start))

    checkpoint = {'args':args, 'model':model.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(os.path.join(save_path, 'checkpoints', '{}'.format(model_path))))
    model.eval()
    val_result = validate_epoch(model=model,
                                metric_func=metric_func,
                                metric_list=metric_list,
                                valid_loader=test_loader,
                                in_seq_len=args.dataset_config['in_seq'],
                                out_seq_len=args.dataset_config['out_seq'],
                                y_normalizer=y_normalizer,
                                device=device,
                                )

    metrics_ave_dict = {}
    for metric_name in metric_list:
        metrics_ave_dict[metric_name] = np.mean(val_result[metric_name])

    metric_log = ''
    for metric_name in metric_list:
        metric_log += 'Val {}: {}; '.format(metric_name, metrics_ave_dict[metric_name])
    print(metric_log)




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


def validate_epoch(model,
                   metric_func,
                   metric_list,
                   valid_loader,
                   in_seq_len,
                   out_seq_len,
                   y_normalizer,
                   device,
                   is_vis=False,
                   sample_dir=None,
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

            # gt = rearrange(g.ndata['y'], 'bn (t c) 1 -> bn t c', t=in_seq_len + out_seq_len, c=2)
            # out = rearrange(out, 'bn (t c) -> bn t c', t=out_seq_len, c=2)
            gt = rearrange(g.ndata['y'], 'bn (c t) 1 -> bn t c', t=in_seq_len + out_seq_len, c=2)
            out = rearrange(out, 'bn (c t) -> bn t c', t=out_seq_len, c=2)

            out_full = gt.clone()
            out_full[:, in_seq_len:in_seq_len + out_seq_len, :] = out
            y_pred = out_full
            y = gt
            y_pred = rearrange(y_pred, '(b n) t c -> b n t c', b=bs)
            y = rearrange(y, '(b n) t c -> b n t c', b=bs)

            y_pred = y_normalizer.transform(y_pred, inverse=True)
            y = y_normalizer.transform(y, inverse=True)

            if is_vis:
                if _ > 0:
                    break
                y_pred_array = y_pred.cpu().numpy().reshape(-1, 128, 128, 101, 2)[0]
                y_array = y.cpu().numpy().reshape(-1, 128, 128, 101, 2)[0]
                result_path = os.path.join(sample_dir)
                mkdir(result_path)
                np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                         pred=y_pred_array,
                         gt=y_array)

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

    args.dataset = 'dr2d'
    is_vis = args.is_vis

    # load weight
    weight_path = args.weight_path

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    timestamp = 'test'
    save_path = os.path.join('results', args.comment + '_' + timestamp)
    mkdir(save_path)

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

    checkpoint = torch.load(weight_path, map_location=device,)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)

    writer = None
    log_path = None

    print(model)
    # print(config)


    time_start = time.time()

    model.eval()
    val_result = validate_epoch(model=model,
                                metric_func=metric_func,
                                metric_list=metric_list,
                                valid_loader=test_loader,
                                in_seq_len=args.dataset_config['in_seq'],
                                out_seq_len=args.dataset_config['out_seq'],
                                y_normalizer=y_normalizer,
                                device=device,
                                is_vis=is_vis,
                                sample_dir=save_path,
                                )

    metrics_ave_dict = {}
    for metric_name in metric_list:
        metrics_ave_dict[metric_name] = np.mean(val_result[metric_name])

    metric_log = ''
    for metric_name in metric_list:
        metric_log += 'Val {}: {}; '.format(metric_name, metrics_ave_dict[metric_name])
    print(metric_log)




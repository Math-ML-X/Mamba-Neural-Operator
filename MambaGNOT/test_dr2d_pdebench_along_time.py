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
from utils_pack.util_metrics import eval_2d_time
import csv

'''
    A general code framework for training neural operator on irregular domains
'''

EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']


def save_metric_time_dicts_to_csv(metric_time_ave_dict, metric_time_std_dict, save_path,
                                  filename='metric_time_results.csv'):
    """
    Save the average and std metric dictionaries to a CSV file.

    Parameters:
        metric_time_ave_dict (dict): Average values, like {'MSE': {'t_0': val, ...}, ...}
        metric_time_std_dict (dict): Std values, same structure as above
        save_path (str): Directory to save the CSV file
        filename (str): Output filename
    """
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, filename)

    # Collect all time steps
    time_steps = sorted(next(iter(metric_time_ave_dict.values())).keys(), key=lambda x: int(x[2:]))

    # Prepare header: e.g., MSE_Mean, MSE_Std, ...
    metrics = list(metric_time_ave_dict.keys())
    header = []
    for metric in metrics:
        header.append(f"{metric}_Mean")
        header.append(f"{metric}_Std")

    # Prepare rows for each time step
    rows = []
    for t in time_steps:
        row = []
        for metric in metrics:
            row.append(metric_time_ave_dict[metric].get(t, ''))
            row.append(metric_time_std_dict[metric].get(t, ''))
        rows.append(row)

    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved metric results to {csv_path}")


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
    metric_time_dict = {}
    for metric_name in metric_list:
        metric_dict[metric_name] = []
        metric_time_dict[metric_name] = {}
        for i in range(in_seq_len + out_seq_len):
            metric_time_dict[metric_name][f't_{i}'] = []

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

            metrics_time = metric_func(y_pred, y, metric_list=metric_list)

            for i in range(in_seq_len + out_seq_len):
                for metric_name in metric_list:
                    metric_time_dict[metric_name][f't_{i}'].append(metrics_time[f't_{i}'][metric_name])

    return metric_time_dict


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
    metric_func = eval_2d_time

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
    val_result_time = validate_epoch(model=model,
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

    metric_time_ave_dict = {}
    metric_time_std_dict = {}
    for metric_name in metric_list:
        metric_time_ave_dict[metric_name] = {}
        metric_time_std_dict[metric_name] = {}
    for i in range(args.dataset_config['in_seq'] + args.dataset_config['out_seq']):
        for metric_name in metric_list:
            metric_time_ave_dict[metric_name][f't_{i}'] = np.mean(val_result_time[metric_name][f't_{i}'])
            metric_time_std_dict[metric_name][f't_{i}'] = np.std(val_result_time[metric_name][f't_{i}'])

    save_metric_time_dicts_to_csv(metric_time_ave_dict=metric_time_ave_dict,
                                  metric_time_std_dict=metric_time_std_dict,
                                  save_path=save_path,
                                  )





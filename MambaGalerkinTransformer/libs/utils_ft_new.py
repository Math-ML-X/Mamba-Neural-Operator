import argparse
import math
import os
import sys
import re
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd
import torch
from matplotlib import rc, rcParams, tri
from numpy.core.numeric import identity
from scipy.io import loadmat
from scipy.sparse import csr_matrix, diags
from scipy.sparse import hstack as sparse_hstack
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# try:
#     from utils import *
# except:
    # from galerkin_transformer.utils import *
from .utils import *
from einops import rearrange, repeat

try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as e:
    print('Please install Plotly for showing mesh and solutions.')

# current_path = os.path.dirname(os.path.abspath(__file__))
# SRC_ROOT = os.path.dirname(current_path)
# MODEL_PATH = default(os.environ.get('MODEL_PATH'),
#                      os.path.join(SRC_ROOT, 'models'))
# DATA_PATH = default(os.environ.get('DATA_PATH'),
#                     os.path.join(SRC_ROOT, 'data'))
# FIG_PATH = default(os.environ.get('FIG_PATH'),
#                    os.path.join(os.path.dirname(SRC_ROOT), 'figures'))
EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']
PI = math.pi
SEED = default(os.environ.get('SEED'), 1127802)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_model_name(model='darcy',
                   attention_type='fourier',
                   additional_str: str = '',
                   ):

    model_name = model + '_'

    if attention_type == 'fourier':
        attn_str = f'ft_'
    elif attention_type == 'galerkin':
        attn_str = f'gt_'
    elif attention_type == 'linear':
        attn_str = f'lt_'
    elif attention_type == 'softmax':
        attn_str = f'st_'
    elif attention_type == 'mamba_c4':
        attn_str = f'mb_c4_'
    elif attention_type == 'mamba':
        attn_str = f'mb_'
    else:
        raise NotImplementedError("Attention type not implemented.")
    model_name += attn_str
    if additional_str:
        model_name += additional_str
    return model_name

def run_train_pdebench(
    model,
    loss_func,
    metric_func,
    metric_list,
    train_loader,
    valid_loader,
    optimizer,
    lr_scheduler,
    in_seq_len=10,
    out_seq_len=91,
    train_batch=None,
    validate_epoch=None,
    y_normalizer=None,
    epochs=10,
    device="cuda",
    mode='min',
    tqdm_mode='batch',
    patience=None,
    grad_clip=0.999,
    start_epoch: int = 0,
    model_save_path='models',
    save_mode='state_dict',  # 'state_dict' or 'entire'
    model_name='model',
):

    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = -np.inf if mode == 'max' else np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__) for s in EPOCH_SCHEDULERS)
    lr_scheduler = None if is_epoch_scheduler else lr_scheduler
    tqdm_epoch = False if tqdm_mode == 'batch' else True

    with tqdm(total=end_epoch-start_epoch, disable=not tqdm_epoch) as pbar_ep:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            with tqdm(total=len(train_loader), disable=tqdm_epoch) as pbar_batch:
                for batch in train_loader:
                    loss = train_batch(
                        model=model,
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
                    desc = f"epoch: [{epoch+1}/{end_epoch}]"
                    if loss.ndim == 0:  # 1 target loss
                        _loss_mean = np.mean(loss_epoch)
                        desc += f" loss: {_loss_mean:.3e}"
                    else:
                        _loss_mean = np.mean(loss_epoch, axis=0)
                        for j in range(len(_loss_mean)):
                            if _loss_mean[j] > 0:
                                desc += f" | loss {j}: {_loss_mean[j]:.3e}"
                    desc += f" | current lr: {lr:.3e}"
                    pbar_batch.set_description(desc)
                    pbar_batch.update()

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

            if mode == 'max':
                if val_metric > best_val_metric:
                    best_val_epoch = epoch
                    best_val_metric = val_metric
                    stop_counter = 0
                else:
                    stop_counter += 1
            else:
                if val_metric < best_val_metric:
                    best_val_epoch = epoch
                    best_val_metric = val_metric
                    stop_counter = 0
                    if save_mode == 'state_dict':
                        torch.save(model.state_dict(), os.path.join(model_save_path, '{}.pt'.format(model_name)))
                    else:
                        torch.save(model, os.path.join(model_save_path, '{}.pt'.format(model_name)))
                    best_model_state_dict = {
                        k: v.to('cpu') for k, v in model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

                else:
                    stop_counter += 1

            if lr_scheduler and is_epoch_scheduler:
                if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                    lr_scheduler.step(val_metric)
                else:
                    lr_scheduler.step()

            if stop_counter > patience:
                print(f"Early stop at epoch {epoch}")
                break

            desc = color(f"| val metric: {val_metric:.3e} ", color=Colors.blue)
            desc += color(f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch+1}",
                          color=Colors.yellow)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.red)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)
            if not tqdm_epoch:
                tqdm.write("\n"+desc+"\n")
            else:
                desc_ep = color("", color=Colors.green)
                if _loss_mean.ndim == 0:  # 1 target loss
                    desc_ep += color(f"| loss: {_loss_mean:.3e} ",
                                     color=Colors.green)
                else:
                    for j in range(len(_loss_mean)):
                        if _loss_mean[j] > 0:
                            desc_ep += color(
                                f"| loss {j}: {_loss_mean[j]:.3e} ", color=Colors.green)
                desc_ep += desc
                pbar_ep.set_description(desc_ep)
                pbar_ep.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_val=np.asarray(loss_val),
                lr_history=np.asarray(lr_history),
                # best_model=best_model_state_dict,
                optimizer_state=optimizer.state_dict()
            )
            save_pickle(result, os.path.join(model_save_path, '{}.pkl'.format(model_name)))
    return result


def train_batch_cns2d_pdebench(model,
                              loss_func,
                              data,
                              optimizer,
                              lr_scheduler,
                              in_seq_len,
                              out_seq_len,
                              device,
                              grad_clip=0.99):
    optimizer.zero_grad()
    u_inseq = data["feat_inseq"].to(device)
    u_query_fullseq = data["feat_query_fullseq"].to(device)
    grid = data["grid"].to(device)

    u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
    u_pred = model(u_inseq, pos=grid)
    u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=4)

    u_pred_full = u_query_fullseq.clone()
    u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

    loss = loss_func(u_pred_full, u_query_fullseq)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item()


def validate_epoch_cns2d_pdebench(model,
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
            u_inseq = data["feat_inseq"].to(device)
            u_query_fullseq = data["feat_query_fullseq"].to(device)
            grid = data["grid"].to(device)

            u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
            u_pred = model(u_inseq, pos=grid)
            u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=4)

            u_pred_full = u_query_fullseq.clone()
            u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

            # u_pred_full = y_normalizer.transform(u_pred_full, inverse=True)
            # u_query_fullseq = y_normalizer.transform(u_query_fullseq, inverse=True)

            if is_vis:
                if _ > 0:
                    break
                u_pred_full_array = u_pred_full.cpu().numpy().reshape(-1, 512, 512, 11, 4)[0]
                u_query_fullseq_array = u_query_fullseq.cpu().numpy().reshape(-1, 512, 512, 11, 4)[0]
                result_path = os.path.join(sample_dir)
                mkdir(result_path)
                np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                         pred=u_pred_full_array,
                         gt=u_query_fullseq_array)

            metrics = metric_func(u_pred_full, u_query_fullseq, metric_list=metric_list)
            for metric_name in metric_list:
                metric_dict[metric_name].append(metrics[metric_name])

    return metric_dict


def train_batch_dr2d_pdebench(model,
                              loss_func,
                              data,
                              optimizer,
                              lr_scheduler,
                              in_seq_len,
                              out_seq_len,
                              device,
                              grad_clip=0.99):
    optimizer.zero_grad()
    u_inseq = data["feat_inseq"].to(device)
    u_query_fullseq = data["feat_query_fullseq"].to(device)
    grid = data["grid"].to(device)

    u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
    u_pred = model(u_inseq, pos=grid)
    u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=2)

    u_pred_full = u_query_fullseq.clone()
    u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

    loss = loss_func(u_pred_full, u_query_fullseq)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item()


def validate_epoch_dr2d_pdebench(model,
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
            u_inseq = data["feat_inseq"].to(device)
            u_query_fullseq = data["feat_query_fullseq"].to(device)
            grid = data["grid"].to(device)

            u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
            u_pred = model(u_inseq, pos=grid)
            u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=2)

            u_pred_full = u_query_fullseq.clone()
            u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

            u_pred_full = y_normalizer.transform(u_pred_full, inverse=True)
            u_query_fullseq = y_normalizer.transform(u_query_fullseq, inverse=True)

            if is_vis:
                if _ > 0:
                    break
                u_pred_full_array = u_pred_full.cpu().numpy().reshape(-1, 128, 128, 101, 2)[0]
                u_query_fullseq_array = u_query_fullseq.cpu().numpy().reshape(-1, 128, 128, 101, 2)[0]
                result_path = os.path.join(sample_dir)
                mkdir(result_path)
                np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                         pred=u_pred_full_array,
                         gt=u_query_fullseq_array)

            metrics = metric_func(u_pred_full, u_query_fullseq, metric_list=metric_list)
            for metric_name in metric_list:
                metric_dict[metric_name].append(metrics[metric_name])

    return metric_dict


def validate_epoch_dr2d_pdebench_along_time(model,
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
            u_inseq = data["feat_inseq"].to(device)
            u_query_fullseq = data["feat_query_fullseq"].to(device)
            grid = data["grid"].to(device)

            u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
            u_pred = model(u_inseq, pos=grid)
            u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=2)

            u_pred_full = u_query_fullseq.clone()
            u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

            u_pred_full = y_normalizer.transform(u_pred_full, inverse=True)
            u_query_fullseq = y_normalizer.transform(u_query_fullseq, inverse=True)

            metrics_time = metric_func(u_pred_full, u_query_fullseq, metric_list=metric_list)

            for i in range(in_seq_len + out_seq_len):
                for metric_name in metric_list:
                    metric_time_dict[metric_name][f't_{i}'].append(metrics_time[f't_{i}'][metric_name])

    return metric_time_dict




def train_batch_sw2d_pdebench(model,
                              loss_func,
                              data,
                              optimizer,
                              lr_scheduler,
                              in_seq_len,
                              out_seq_len,
                              device,
                              grad_clip=0.99):
    optimizer.zero_grad()
    u_inseq = data["feat_inseq"].to(device)
    u_query_fullseq = data["feat_query_fullseq"].to(device)
    grid = data["grid"].to(device)

    u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
    u_pred = model(u_inseq, pos=grid)
    u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=1)

    u_pred_full = u_query_fullseq.clone()
    u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

    loss = loss_func(u_pred_full, u_query_fullseq)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item()


def validate_epoch_sw2d_pdebench(model,
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
            u_inseq = data["feat_inseq"].to(device)
            u_query_fullseq = data["feat_query_fullseq"].to(device)
            grid = data["grid"].to(device)

            u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
            u_pred = model(u_inseq, pos=grid)
            u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=1)

            u_pred_full = u_query_fullseq.clone()
            u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

            u_pred_full = y_normalizer.transform(u_pred_full, inverse=True)
            u_query_fullseq = y_normalizer.transform(u_query_fullseq, inverse=True)

            if is_vis:
                if _ > 0:
                    break
                u_pred_full_array = u_pred_full.cpu().numpy().reshape(-1, 128, 128, 101, 1)[0]
                u_query_fullseq_array = u_query_fullseq.cpu().numpy().reshape(-1, 128, 128, 101, 1)[0]
                result_path = os.path.join(sample_dir)
                mkdir(result_path)
                np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                         pred=u_pred_full_array,
                         gt=u_query_fullseq_array)

            metrics = metric_func(u_pred_full, u_query_fullseq, metric_list=metric_list)
            for metric_name in metric_list:
                metric_dict[metric_name].append(metrics[metric_name])

    return metric_dict


def validate_epoch_sw2d_pdebench_along_time(model,
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
            u_inseq = data["feat_inseq"].to(device)
            u_query_fullseq = data["feat_query_fullseq"].to(device)
            grid = data["grid"].to(device)

            u_inseq = rearrange(u_inseq, 'b n t a -> b n (t a)')
            u_pred = model(u_inseq, pos=grid)
            u_pred = rearrange(u_pred, 'b n (t a) -> b n t a', t=out_seq_len, a=1)

            u_pred_full = u_query_fullseq.clone()
            u_pred_full[:, :, in_seq_len:in_seq_len + out_seq_len, :] = u_pred

            u_pred_full = y_normalizer.transform(u_pred_full, inverse=True)
            u_query_fullseq = y_normalizer.transform(u_query_fullseq, inverse=True)

            metrics_time = metric_func(u_pred_full, u_query_fullseq, metric_list=metric_list)

            for i in range(in_seq_len + out_seq_len):
                for metric_name in metric_list:
                    metric_time_dict[metric_name][f't_{i}'].append(metrics_time[f't_{i}'][metric_name])

    return metric_time_dict


def train_batch_darcyflow_pdebench(model,
                                   loss_func,
                                   data,
                                   optimizer,
                                   lr_scheduler,
                                   in_seq_len,
                                   out_seq_len,
                                   device,
                                   grad_clip=0.99):
    optimizer.zero_grad()
    a = data["a"].to(device)
    u = data["u"].to(device)
    grid = data["grid"].to(device)

    u_pred = model(a, pos=grid)

    loss = loss_func(u_pred, u)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item()


def validate_epoch_darcyflow_pdebench(model,
                                      metric_func,
                                      metric_list,
                                      valid_loader,
                                      y_normalizer,
                                      in_seq_len,
                                      out_seq_len,
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
            a = data["a"].to(device)
            u = data["u"].to(device)
            grid = data["grid"].to(device)

            u_pred = model(a, pos=grid)

            if is_vis:
                if _ > 0:
                    break
                a_array = a.cpu().numpy().reshape(-1, 64, 64, 1)[0]
                u_pred_array = u_pred.cpu().numpy().reshape(-1, 64, 64, 1)[0]
                u_array = u.cpu().numpy().reshape(-1, 64, 64, 1)[0]
                result_path = os.path.join(sample_dir)
                mkdir(sample_dir)
                np.savez(os.path.join(result_path, '{}.npz'.format("vis_results")),
                         input=a_array,
                         pred=u_pred_array,
                         gt=u_array)

            metrics = metric_func(u_pred, u, metric_list=metric_list)
            for metric_name in metric_list:
                metric_dict[metric_name].append(metrics[metric_name])

    return metric_dict




if __name__ == '__main__':
    get_seed(42)

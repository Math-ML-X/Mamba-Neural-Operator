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
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl

from utils_pack.util_fourier_analysis import *

'''
    A general code framework for training neural operator on irregular domains
'''

EPOCH_SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'MultiplicativeLR',
                    'MultiStepLR', 'ExponentialLR', 'LambdaLR']

def get_specific_layer_output(x_in, model, layer_name_list, transform=None):
    hook_outputs = []
    def hook(module, input, output):
        hook_outputs.append(output)

    hook_handle_list = []
    for name, module in model.named_modules():
        if name in layer_name_list:
            hook_handle = module.register_forward_hook(hook)
            hook_handle_list.append(hook_handle)

    if transform is not None:
        x_in = transform(x_in)

    x_out = model(*x_in)

    for hook_handle in hook_handle_list:
        hook_handle.remove()

    return hook_outputs



def validate_epoch(model,
                   metric_func,
                   metric_list,
                   valid_loader,
                   in_seq_len,
                   out_seq_len,
                   y_normalizer,
                   layer_name_list,
                   save_path,
                   device,
                   ):
    model.eval()
    metric_dict = {}
    for metric_name in metric_list:
        metric_dict[metric_name] = []
    for _, data in enumerate(valid_loader):

        if _ > 0:
            break
        with torch.no_grad():
            g, u_p, g_u = data
            g, g_u, u_p = g.to(device), g_u.to(device), u_p.to(device)

            bs = g.batch_size


            latents = get_specific_layer_output(x_in=(g, u_p, g_u),
                                               model=model,
                                               layer_name_list=layer_name_list,
                                               transform=None)

            # out = model(g, u_p, g_u)

            # Fourier transform feature maps
            fourier_latents = []
            for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
                latent = latent.cpu()

                if len(latent.shape) == 3:  # for ViT
                    b, n, c = latent.shape
                    h, w = int(math.sqrt(n)), int(math.sqrt(n))
                    latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
                elif len(latent.shape) == 4:  # for CNN
                    b, c, h, w = latent.shape
                else:
                    raise Exception("shape: %s" % str(latent.shape))
                latent = fourier(latent)
                latent = shift(latent).mean(dim=(0, 1))
                latent = latent.diag()[int(h / 2):]  # only use the half-diagonal components
                latent = latent - latent[0]  # visualize 'relative' log amplitudes
                # (i.e., low-freq amp - high freq amp)
                fourier_latents.append(latent)

        # A. Plot Fig 2a: "Relative log amplitudes of Fourier transformed feature maps"
        fig, ax1 = plt.subplots(1, 1, figsize=(3.3, 4), dpi=150)
        for i, latent in enumerate(reversed(fourier_latents[:-1])):
            freq = np.linspace(0, 1, len(latent))
            ax1.plot(freq, latent, color=cm.plasma_r(i / len(fourier_latents)))

        ax1.set_xlim(left=0, right=1)

        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("$\Delta$ Log amplitude")

        from matplotlib.ticker import FormatStrFormatter
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

        # ---- 新增：给图1加colorbar（颜色映射层/深度索引）----
        mappable1 = mpl.cm.ScalarMappable(
            cmap=cm.plasma_r, norm=mpl.colors.Normalize(vmin=0, vmax=1)
        )
        mappable1.set_array([])
        cbar1 = fig.colorbar(mappable1, ax=ax1, fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar1.set_label("Layer index (shallow→deep)")
        # -------------------------------------------------------

        # save ax1 fig use plt
        fig.savefig(os.path.join(save_path, 'fourier_transform_feature_maps.png'))
        plt.close(fig)

        pools = []
        msas = [1, 3, 5, 7, 9, 11, 13, 15, ]
        marker = "o"

        depths = range(len(fourier_latents))

        # Normalize
        depth = len(depths) - 1
        depths = (np.array(depths)) / depth
        pools = (np.array(pools)) / depth
        msas = (np.array(msas)) / depth

        fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 4), dpi=120)
        plot_segment(ax2, depths, [latent[-1] for latent in fourier_latents],
                     marker=marker)  # high-frequency component

        for pool in pools:
            ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
        for msa in msas:
            ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)

        ax2.set_xlabel("Normalized depth")
        ax2.set_ylabel("$\Delta$ Log amplitude")
        ax2.set_xlim(0.0, 1.0)

        from matplotlib.ticker import FormatStrFormatter
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # plt.show()
        # ---- 新增：给图2加colorbar（颜色映射归一化深度）----
        mappable2 = mpl.cm.ScalarMappable(
            cmap=cm.plasma_r, norm=mpl.colors.Normalize(vmin=0, vmax=1)
        )
        mappable2.set_array([])
        cbar2 = fig.colorbar(mappable2, ax=ax2, fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar2.set_label("Normalized depth (0→1)")
        # ------------------------------------------------------

        # save ax2 fig use plt
        fig.savefig(os.path.join(save_path, 'fourier_transform_feature_maps_normalized.png'))
        plt.close(fig)

            # y_pred = out
            # y = g.ndata['y']
            #
            # y_pred = rearrange(y_pred, '(b n) c -> b n c', b=bs)
            # y = rearrange(y, '(b n) c -> b n c', b=bs)
            #
            # metrics = metric_func(y_pred, y, metric_list=metric_list)
            # for metric_name in metric_list:
            #     metric_dict[metric_name].append(metrics[metric_name])

    return metric_dict


if __name__ == "__main__":

    args = get_args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    args.dataset = 'darcyflow2d'

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
    y_normalizer = None

    get_seed(args.seed)
    torch.cuda.empty_cache()

    loss_func = nn.MSELoss()
    metric_list = ['RMSE', 'nRMSE', 'Rel_L2_Norm']

    layer_name_list = ['trunk_mlp',
                       'blocks.0.crossattn', 'blocks.0.moe_mlp1',
                       'blocks.0.selfattn', 'blocks.0.moe_mlp2',
                       'blocks.1.crossattn', 'blocks.1.moe_mlp1',
                       'blocks.1.selfattn', 'blocks.1.moe_mlp2',
                       'blocks.2.crossattn', 'blocks.2.moe_mlp1',
                       'blocks.2.selfattn', 'blocks.2.moe_mlp2',
                       'blocks.3.crossattn', 'blocks.3.moe_mlp1',
                       'blocks.3.selfattn', 'blocks.3.moe_mlp2',
                       'out_mlp']

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
                                in_seq_len=None,
                                out_seq_len=None,
                                y_normalizer=y_normalizer,
                                layer_name_list=layer_name_list,
                                save_path=save_path,
                                device=device,
                                )

    metrics_ave_dict = {}
    for metric_name in metric_list:
        metrics_ave_dict[metric_name] = np.mean(val_result[metric_name])

    metric_log = ''
    for metric_name in metric_list:
        metric_log += 'Val {}: {}; '.format(metric_name, metrics_ave_dict[metric_name])
    print(metric_log)




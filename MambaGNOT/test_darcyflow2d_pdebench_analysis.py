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

import dgl
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


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count
    args = get_args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")

    get_seed(args.seed, printout=False)


    _, test_dataset = get_dataset_new(args)
    test_loader = MIODataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False)

    args.space_dim = 2

    get_seed(args.seed)
    torch.cuda.empty_cache()

    model = get_model(args).to(device)

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    # ========================================
    # 构造 dummy 输入用于 forward_from_tensor
    # ========================================
    g, u_p, g_u = next(iter(test_loader))

    g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

    gs = dgl.unbatch(g)
    x_raw = nn.utils.rnn.pad_sequence([_g.ndata['x'] for _g in gs]).permute(1, 0, 2)  # B, T, C
    num_nodes = [g_.num_nodes() for g_ in gs]
    inputs = g_u[0]

    # ========================================
    # FLOPs 和 Params 统计
    # ========================================
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (x_raw, u_p, inputs, num_nodes))
        flops_total = flops.total() / 1e9  # GFlops

        param_count_dict = parameter_count(model)
        params_total = sum(param_count_dict.values()) / 1e6  # M

        print(f"[fvcore] Total Parameters: {params_total:.2f} M")
        print(f"[fvcore] Total FLOPs: {flops_total:.2f} G")

    # ========================================
    # Inference 计时
    # ========================================
    num_iter = 100
    elapsed_time_list = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for i in range(num_iter):
        g, u_p, g_u = next(iter(test_loader))
        g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

        gs = dgl.unbatch(g)
        x_raw = nn.utils.rnn.pad_sequence([_g.ndata['x'] for _g in gs]).permute(1, 0, 2)
        num_nodes = [g_.num_nodes() for g_ in gs]
        inputs = g_u[0]

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model.forward(x_raw, u_p, inputs, num_nodes)
        torch.cuda.synchronize()
        elapsed_time_list.append(time.time() - start_time)

        if i == 0:
            # 只记录第一个推理的最大显存占用（GiB）
            max_memory_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"[Info] Max GPU Memory Usage (1st run): {max_memory_gb:.2f} GiB")

    avg_infer_time = np.mean(elapsed_time_list)
    print(f"[Info] Average Inference Time over {num_iter} runs: {avg_infer_time * 1000:.2f} ms")

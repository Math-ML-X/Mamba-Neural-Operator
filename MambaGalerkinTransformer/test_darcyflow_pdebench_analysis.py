from libs_path import *
from libs import *
from utils_pack.util_common import *
from utils_pack.util_metrics import eval_2d
from dataset.dataset_darcy_flow_pdebench_d3 import DarcyFlowDataset
import argparse
from datetime import datetime


def main(config_path='config/darcyflow_pdebench/config_gt_darcyflow_pdebench.yml'):
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('--config-path', type=str, default=config_path)
    parser.add_argument('--weight-path', type=str)
    parser.add_argument('--is_vis', action='store_true', help='if output samples')

    config_path = parser.parse_args().config_path
    weight_path = parser.parse_args().weight_path
    is_vis = parser.parse_args().is_vis

    with open(config_path) as f:
        config = yaml.full_load(f)
    config = config['darcyflow_pdebench']

    seed = config['seed']
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(seed)

    train_dataset = None
    valid_dataset = DarcyFlowDataset(dataset_params=config['dataset'], split='test')

    train_loader = None
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['dataset']['test']['batch_size'],
                              shuffle=False,
                              drop_last=False,
                              **kwargs)

    config['attn_norm'] = not config['layer_norm']
    if config['attention_type'] == 'fourier':
        config['norm_eps'] = 1e-7
    elif config['attention_type'] == 'galerkin':
        config['norm_eps'] = 1e-5

    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.empty_cache()
    model = FourierTransformerNew(**config)

    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    # ========== Measure Params, FLOPs, Inference Time ==========
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    import torch.nn as nn
    import time

    # 包一层 Wrapper 让 grid 变为位置参数，适配 fvcore
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, a, grid):
            return self.model(a, pos=grid)

    model_wrapped = WrappedModel(model).to(device)

    model_wrapped.eval()
    with torch.no_grad():
        first_batch = next(iter(valid_loader))
        a = first_batch["a"].to(device)  # [B, C, H, W]
        grid = first_batch["grid"].to(device)  # [B, H, W, 2]

        # ---------- Params ----------
        param_counts = parameter_count(model_wrapped)
        total_params = sum(param_counts.values())
        print(f"[Params - fvcore] Total: {total_params / 1e6:.2f} M")

        # print("[Parameter Table]")
        # for name, val in param_counts.items():
        #     print(f"  {name or 'total'}: {val / 1e6:.2f} M")

        # ---------- FLOPs ----------
        try:
            flops = FlopCountAnalysis(model_wrapped, (a, grid))
            total_flops = flops.total()
            print(f"[FLOPs - fvcore] Total: {total_flops / 1e9:.2f} GFLOPs")
            print("[FLOPs breakdown]")
            # print(flop_count_table(flops, max_depth=2))
        except Exception as e:
            print(f"[FLOPs] Failed to compute FLOPs with fvcore: {e}")

        # ---------- Inference Time ----------
        timings = []

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        for i in range(100):
            start_time = time.time()
            _ = model(a, pos=grid)  # 使用原始模型，不加 wrapper
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)

            if i == 0:
                max_memory_bytes = torch.cuda.max_memory_allocated(device)
                max_memory_gib = max_memory_bytes / (1024 ** 3)
                print(f"[Max GPU Memory Usage] 1st run: {max_memory_gib:.2f} GiB")

        avg_time = sum(timings) / len(timings)
        print(f"[Inference Time] 100 rounds average: {avg_time * 1000:.2f} ms")


if __name__ == '__main__':
    main()
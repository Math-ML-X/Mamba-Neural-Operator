from libs_path import *
from libs import *
from utils_pack.util_common import *
from utils_pack.util_metrics import eval_2d
from dataset.dataset_shallow_water_2d_pdebench_d3 import ShallowWater2DDataset
import argparse
from datetime import datetime


def main(config_path='config/sw2d_pdebench/config_gt_sw2d_pdebench.yml'):
    parser = argparse.ArgumentParser(description='Configurations')
    parser.add_argument('--config-path', type=str, default=config_path)
    parser.add_argument('--weight-path', type=str)
    parser.add_argument('--is_vis', action='store_true', help='if output samples')

    config_path = parser.parse_args().config_path
    weight_path = parser.parse_args().weight_path
    is_vis = parser.parse_args().is_vis

    with open(config_path) as f:
        config = yaml.full_load(f)
    config = config['sw2d_pdebench']

    seed = config['seed']
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(seed)

    train_dataset = None
    valid_dataset = ShallowWater2DDataset(dataset_params=config['dataset'], split='test')

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

    checkpoint = torch.load(weight_path, map_location=device,)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    model_name = get_model_name(model='sw2d_pdebench',
                                attention_type=config['attention_type'],
                                additional_str=config['additional_str'],
                                )

    save_path = config['save_path']
    time_stamp = 'test'
    save_path = os.path.join(save_path, "{}_{}".format(model_name, time_stamp))
    mkdir(save_path)

    in_seq_len = config['in_seq_len']
    out_seq_len = config['out_seq_len']

    metric_list = ['RMSE', 'nRMSE', 'Rel_L2_Norm']
    metric_func = eval_2d

    model.eval()
    val_result = validate_epoch_sw2d_pdebench(model=model,
                                              metric_func=metric_func,
                                              metric_list=metric_list,
                                              valid_loader=valid_loader,
                                              in_seq_len=in_seq_len,
                                              out_seq_len=out_seq_len,
                                              y_normalizer=valid_dataset.y_normalizer,
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



if __name__ == '__main__':
    # main("config/sw2d_pdebench/config_gt_sw2d_pdebench.yml")
    main("config/sw2d_pdebench/config_mamba_sw2d_pdebench.yml")


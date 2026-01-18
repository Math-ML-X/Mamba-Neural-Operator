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
    config_path = parser.parse_args().config_path

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

    train_dataset = ShallowWater2DDataset(dataset_params=config['dataset'], split='train')
    valid_dataset = ShallowWater2DDataset(dataset_params=config['dataset'], split='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=config['dataset']['train']['batch_size'],
                              shuffle=True,
                              drop_last=True,
                              **kwargs)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['dataset']['test']['batch_size'],
                              shuffle=False,
                              drop_last=False,
                              **kwargs)

    sample = next(iter(train_loader))

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))

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

    model_name = get_model_name(model='sw2d_pdebench',
                                attention_type=config['attention_type'],
                                additional_str=config['additional_str'],
                                )

    save_path = config['save_path']
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_path, "{}_{}".format(model_name, time_stamp))
    mkdir(save_path)

    print(f"Saving model and result in {save_path}...\n")

    epochs = config['epochs']
    lr = config['lr']
    in_seq_len = config['in_seq_len']
    out_seq_len = config['out_seq_len']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                           div_factor=1e4,
                           final_div_factor=1e4,
                           pct_start=0.3,
                           steps_per_epoch=len(train_loader),
                           epochs=epochs)

    loss_func = nn.MSELoss()
    metric_list = ['RMSE', 'nRMSE', 'Rel_L2_Norm']
    metric_func = eval_2d

    result = run_train_pdebench(model=model,
                                loss_func=loss_func,
                                metric_func=metric_func,
                                metric_list=metric_list,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                optimizer=optimizer,
                                lr_scheduler=scheduler,
                                in_seq_len=in_seq_len,
                                out_seq_len=out_seq_len,
                                train_batch=train_batch_sw2d_pdebench,
                                validate_epoch=validate_epoch_sw2d_pdebench,
                                y_normalizer=valid_dataset.y_normalizer,
                                epochs=epochs,
                                device=device,
                                model_save_path=save_path,
                                model_name=model_name,
                                )

    model.load_state_dict(torch.load(os.path.join(save_path, '{}.pt'.format(model_name))))
    model.eval()

    val_result = validate_epoch_sw2d_pdebench(model=model,
                                              metric_func=metric_func,
                                              metric_list=metric_list,
                                              valid_loader=valid_loader,
                                              in_seq_len=in_seq_len,
                                              out_seq_len=out_seq_len,
                                              y_normalizer=valid_dataset.y_normalizer,
                                              device=device,
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
    # main("config/sw2d_pdebench/config_mamba_sw2d_pdebench.yml")
    main("config/sw2d_pdebench/config_st_sw2d_pdebench_stanx2.yml")

train_dict = {}
train_dict['dataset_path'] = "./PDE/data/DarcyFlow/PDEBench/2D_DarcyFlow_beta1.0_Train/train"
# train_dict['dataset_path'] = "/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/2D_DarcyFlow_beta1.0_Train/train"
train_dict['n_all_samples'] = 9000  # 9000
train_dict['resolution'] = 64
train_dict['reduced_resolution'] = 2
train_dict['reduced_batch'] = 1

test_dict = {}
test_dict['dataset_path'] = "./PDE/data/DarcyFlow/PDEBench/2D_DarcyFlow_beta1.0_Train/test"
# test_dict['dataset_path'] = "/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/2D_DarcyFlow_beta1.0_Train/test"
test_dict['n_all_samples'] = 1000  # 1000
test_dict['resolution'] = 64
test_dict['reduced_resolution'] = 2
test_dict['reduced_batch'] = 1

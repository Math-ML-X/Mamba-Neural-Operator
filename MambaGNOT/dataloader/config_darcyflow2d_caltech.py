train_dict = {}
# train_dict['dataset_path'] = "/home/ubuntu/volume/data/PDE/data/DarcyFlow/Caltech/npz/Darcy_421/train"
train_dict['dataset_path'] = "/media/ssd/data_temp/PDE/data/DarcyFlow/Caltech/npz/Darcy_421/train"
train_dict['n_all_samples'] = 1024  # 9000
train_dict['resolution'] = 141
train_dict['reduced_resolution'] = 3
train_dict['reduced_batch'] = 1
train_dict['real_space_range'] = [0, 1]

test_dict = {}
# test_dict['dataset_path'] = "/home/ubuntu/volume/data/PDE/data/DarcyFlow/Caltech/npz/Darcy_421/test"
test_dict['dataset_path'] = "/media/ssd/data_temp/PDE/data/DarcyFlow/Caltech/npz/Darcy_421/test"
test_dict['n_all_samples'] = 100  # 100
test_dict['resolution'] = 141
test_dict['reduced_resolution'] = 3
test_dict['reduced_batch'] = 1
test_dict['real_space_range'] = [0, 1]
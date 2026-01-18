import os

import numpy as np
from math import ceil
import torch
from torch.utils.data import Dataset
from glob import glob
from einops import rearrange, repeat

from utils_pack.util_mesh import CustomCoodGenerator


class DiffusionReaction2DDataset(Dataset):
    """
    Dataset: Diffusion Reaction 2D Dataset
    Source: https://doi.org/10.18419/darus-2986
    This is a folder dataset (preprocessed).
    Folder:
    - Files (*.npz):
        - Sample (x1)

    """
    def __init__(self, dataset_params, split):
        # load configuration
        self.split = split  # train, val, test
        dataset_params = dataset_params[split]
        self.dataset_params = dataset_params

        # path to dataset folder
        data_folder_path = dataset_params['dataset_path']

        # dataset number and split configuration
        self.n_all_samples = dataset_params['n_all_samples']

        # resolution and domain range configuration
        self.reduced_resolution = dataset_params['reduced_resolution']
        self.reduced_resolution_t = dataset_params['reduced_resolution_t']
        self.reduced_batch = dataset_params['reduced_batch']

        # task specific parameters
        self.in_seq = dataset_params['in_seq']
        self.out_seq = dataset_params['out_seq']

        self.normalize_y = dataset_params['normalize_y']
        self.y_normalizer = None

        # load data list
        self.data_paths_all = sorted(glob(os.path.join(data_folder_path, '*.npz')))
        if self.n_all_samples != len(self.data_paths_all):
            print("Warning: n_all_samples is not equal to the number of files in the folder")
        self.data_paths_all = self.data_paths_all[:self.n_all_samples]

        # dataset has been split into train, test (folder)
        self.n_samples = self.n_all_samples
        self.data_paths = self.data_paths_all

        # load a sample
        self._load_sample(self.data_paths[0])

    def _prepare(self):

        # for one sample
        t = self.grid_t
        u = self.u
        v = self.v

        # set the value & key position (encoder)
        u_sampled = u.clone()
        v_sampled = v.clone()

        u_sampled = rearrange(u_sampled, 't x y a -> x y t a')
        v_sampled = rearrange(v_sampled, 't x y a -> x y t a')

        # undersample on time and space
        u_sampled = u_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        v_sampled = v_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        t_sampled = t[::self.reduced_resolution_t]

        u_sampled = rearrange(u_sampled, 'x y t 1 -> (x y) t 1')
        v_sampled = rearrange(v_sampled, 'x y t 1-> (x y) t 1')

        grid = self.meshgenerator.get_grid()  # (n_sample, 2)

        u_sampled_inseq = u_sampled[:, :self.in_seq, :]
        v_sampled_inseq = v_sampled[:, :self.in_seq, :]

        feat = torch.cat((u_sampled_inseq, v_sampled_inseq), dim=-1)

        # set the query position (additional encoder)
        u_query_sampled = u.clone()
        v_query_sampled = v.clone()

        u_query_sampled = rearrange(u_query_sampled, 't x y a -> x y t a')
        v_query_sampled = rearrange(v_query_sampled, 't x y a -> x y t a')

        # undersample on time and space
        u_query_sampled = u_query_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        v_query_sampled = v_query_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        t_query_sampled = t[::self.reduced_resolution_t]

        u_query_sampled = rearrange(u_query_sampled, 'x y t 1 -> (x y) t 1')
        v_query_sampled = rearrange(v_query_sampled, 'x y t 1-> (x y) t 1')

        grid_query = self.meshgenerator_query.get_grid()  # (n_sample, 2)

        feat_query = torch.cat((u_query_sampled, v_query_sampled), dim=-1)

        self.data_dict = {
            'feat_inseq': feat,
            'feat_query_fullseq': feat_query,
            'grid': grid_query,
        }

    def _load_sample(self, data_path):

        # load data
        with np.load(data_path) as f:

            # self.data = torch.from_numpy(f['data'][()].astype(np.float32))
            self.data = torch.from_numpy(f['data'][()])  # (res_t, res_x, res_y, 2)

            self.u = self.data[:, :, :, 0:1]
            self.v = self.data[:, :, :, 1:2]
            self.uv_mean = torch.from_numpy(f['data_mean'].astype(np.float32))
            self.uv_std = torch.from_numpy(f['data_std'].astype(np.float32))
            self.grid_t = torch.from_numpy(f['grid_t'][()])  # res_t = 101
            self.grid_x = torch.from_numpy(f['grid_x'][()])  # res_x = 128
            self.grid_y = torch.from_numpy(f['grid_y'][()])  # res_y = 128

        data_res_t, data_res_x, data_res_y, _ = self.data.shape  # (res_t, res_x, res_y, 2)
        assert data_res_x == data_res_y
        assert _ == 2
        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = ceil(self.data_res_t / self.reduced_resolution_t)
        assert self.in_seq + self.out_seq == self.res_time

        # mesh grid
        self.grid_x_sampled = self.grid_x[::self.reduced_resolution]
        self.grid_y_sampled = self.grid_y[::self.reduced_resolution]
        gx, gy = torch.meshgrid(self.grid_x_sampled, self.grid_y_sampled)
        self.grid = torch.stack((gy, gx), dim=-1).reshape(-1, 2)  # (res_x * res_y, 2)
        self.meshgenerator = CustomCoodGenerator(grid=self.grid.numpy())

        # mesh grid query
        self.grid_x_query_sampled = self.grid_x[::self.reduced_resolution]
        self.grid_y_query_sampled = self.grid_y[::self.reduced_resolution]
        gx, gy = torch.meshgrid(self.grid_x_query_sampled, self.grid_y_query_sampled)
        self.grid_query = torch.stack((gy, gx), dim=-1).reshape(-1, 2)  # (res_x * res_y, 2)
        self.meshgenerator_query = CustomCoodGenerator(grid=self.grid_query.numpy())

        self._prepare()

        if self.normalize_y:
            self.__normalize_y()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]
        self._load_sample(data_path)

        return self.data_dict

    def __normalize_y(self):
        if self.y_normalizer is None:
            if self.normalize_y == 'unit':
                self.y_normalizer = UnitTransformerWithMeanStd(mean=self.uv_mean, std=self.uv_std)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.mean, self.y_normalizer.std)

        self.data_dict['feat_inseq'] = self.y_normalizer.transform(self.data_dict['feat_inseq'], inverse=False)
        self.data_dict['feat_query_fullseq'] = self.y_normalizer.transform(self.data_dict['feat_query_fullseq'], inverse=False)


class UnitTransformerWithMeanStd():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True, component='all'):
        if component == 'all' or 'all-reduce':
            self.mean = self.mean.to(X.device)
            self.std = self.std.to(X.device)
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            raise NotImplementedError

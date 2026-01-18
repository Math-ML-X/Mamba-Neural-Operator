import os

import numpy as np
from math import ceil
import torch
from torch.utils.data import Dataset
from glob import glob
from einops import rearrange, repeat

from utils_pack.util_mesh import SquareMeshGenerator


class DarcyFlowDataset(Dataset):
    """
    Dataset: Darcy Flow Dataset (PDE Bench)
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
        self.resolution = dataset_params['resolution']
        self.reduced_resolution = dataset_params['reduced_resolution']
        self.reduced_batch = dataset_params['reduced_batch']

        # load data list
        self.data_paths_all = sorted(glob(os.path.join(data_folder_path, '*.npz')))
        if self.n_all_samples != len(self.data_paths_all):
            print("Warning: n_all_samples is not equal to the number of files in the folder")
        self.data_paths_all = self.data_paths_all[:self.n_all_samples]

        # dataset has been split into train, test (folder)
        self.n_samples = self.n_all_samples
        self.data_paths = self.data_paths_all

    def _prepare(self):

        # for one sample
        a = self.a
        u = self.u

        # set the value & key position (encoder)
        a_sampled = a.clone()
        u_sampled = u[0, ...]

        # undersample on time and space
        a_sampled = a_sampled[::self.reduced_resolution, ::self.reduced_resolution]
        u_sampled = u_sampled[::self.reduced_resolution, ::self.reduced_resolution]

        a_sampled = rearrange(a_sampled, 'x y -> (x y) 1')
        u_sampled = rearrange(u_sampled, 'x y -> (x y) 1')

        grid = self.meshgenerator.get_grid()  # (n_sample, 2)

        # set the query position (additional encoder)
        a_query_sampled = a.clone()
        u_query_sampled = u[0, ...]

        # undersample on time and space
        a_query_sampled = a_query_sampled[::self.reduced_resolution, ::self.reduced_resolution]
        u_query_sampled = u_query_sampled[::self.reduced_resolution, ::self.reduced_resolution]

        a_query_sampled = rearrange(a_query_sampled, 'x y -> (x y) 1')
        u_query_sampled = rearrange(u_query_sampled, 'x y -> (x y) 1')

        grid_query = self.meshgenerator_query.get_grid()  # (n_sample, 2)

        # add to the graph
        self.data_dict = {
            'a': a_sampled,
            'u': u_query_sampled,
            'grid': grid_query,
        }

    def _load_sample(self, data_path):

        # load data
        # Keys: ['nu', 'tensor', 'x-coordinate', 'y-coordinate']
        with np.load(data_path) as f:
            self.a = torch.from_numpy(f['nu'])  # (res_x, res_y)
            self.u = torch.from_numpy(f['tensor'])  # (1, res_x, res_y)
            self.x_co = torch.from_numpy(f['x-coordinate'])  # (res_x,)
            self.y_co = torch.from_numpy(f['y-coordinate'])  # (res_y,)

        self.real_space = [[self.x_co[0], self.x_co[-1]],
                           [self.y_co[0], self.y_co[-1]]]

        _, data_res_x, data_res_y, = self.u.shape  # (res_x, res_y, 2)
        assert data_res_x == data_res_y
        assert _ == 1
        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution

        # generate mesh
        self.meshgenerator = SquareMeshGenerator(real_space=self.real_space,
                                                 mesh_size=self.mesh_size,
                                                 downsample_rate=self.reduced_resolution,
                                                 is_diag=False)

        # generate query mesh
        self.meshgenerator_query = SquareMeshGenerator(real_space=self.real_space,
                                                       mesh_size=self.mesh_size,
                                                       downsample_rate=self.reduced_resolution,
                                                       is_diag=False)

        self._prepare()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]
        self._load_sample(data_path)

        return self.data_dict


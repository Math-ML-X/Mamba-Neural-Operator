import torch
import numpy as np
import h5py

class H5Reader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(H5Reader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None

        self._load_file()

    def _load_file(self):
        self.data = h5py.File(self.file_path)

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field][()]

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch
                                     
    def set_float(self, to_float):
        self.to_float = to_float


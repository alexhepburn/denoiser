import torch

import h5py

import numpy as np

from skimage import io
from skimage.color import rgb2gray

from glob import glob

class h5Dataset(torch.utils.data.Dataset):
    """Taken from 
    https://github.com/LabForComputationalVision/bias_free_denoising/blob/master/utils/data/__init__.py"""
    def __init__(self, filename):
        super().__init__()
        self.h5f = h5py.File(filename, "r")
        self.keys = list(self.h5f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, data_size=(1, 50, 50), min=0.0, max=1.0):
        super().__init__()
        self.distribution = torch.distributions.uniform.Uniform(low=min, high=max)
        self.data_size = data_size

    def __len__(self):
        return 403200

    def __getitem__(self, index):
        data = self.distribution.sample(sample_shape=torch.Size(self.data_size))
        return data

class Kodak(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.ims = []
        for f in glob(dir+'/*.png'):
            self.ims.append(rgb2gray(io.imread(f)).astype(np.float32))

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        im = self.ims[index]
        pytorch_im = torch.from_numpy(im).unsqueeze(0)
        return pytorch_im
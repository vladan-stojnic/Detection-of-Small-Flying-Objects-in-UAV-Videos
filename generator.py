# Copyright (c) 2021 Project Bee4Exp.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from keras.utils import Sequence
import h5py
import numpy as np


class HDF5Generator(Sequence):
    """Data generator that uses HDF5 datasets.

    Attributes:
        dataset_path: A path to HDF5 dataset.
        batch_size: An integer size of batch to generate.
        shuffle: A boolean indicating whether to shuffle the dataset.
    """
    def __init__(self, dataset_path, batch_size=64, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = h5py.File(dataset_path, 'r')
        self.data_dset = self.dataset['data']
        self.mask_dset = self.dataset['mask']
        self.on_epoch_end()

    def __len__(self):
        return self.dataset.attrs['elems'] // self.batch_size

    def __getitem__(self, index):
        data = self.data_dset[self.indexes[index]:self.indexes[index]+self.batch_size, :, :, :].copy()

        mask = self.mask_dset[self.indexes[index]:self.indexes[index]+self.batch_size, :, :].copy()
        mask = np.expand_dims(mask, axis=3)

        return data-0.5, mask

    def on_epoch_end(self):
        self.indexes = np.arange(0, self.dataset.attrs['elems'], self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.indexes)

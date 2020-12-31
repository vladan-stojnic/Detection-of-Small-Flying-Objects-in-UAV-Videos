from keras.utils import Sequence
from keras.utils import to_categorical
import h5py
import numpy as np

class HDF5Generator(Sequence):
    def __init__(self, dataset_path, batch_size = 64, shuffle = True, n_classes = 4, out_channels = 1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.dataset = h5py.File(dataset_path, 'r')
        self.data_dset = self.dataset['data']
        self.label_dset = self.dataset['label']
        self.mask_dset = self.dataset['mask']
        self.out_channels = out_channels
        self.on_epoch_end()
        
    def __len__(self):
        return self.dataset.attrs['elems'] // self.batch_size
        
    def __getitem__(self, index):
        data = self.data_dset[self.indexes[index]:self.indexes[index]+self.batch_size, :, :, :].copy()
        label = self.label_dset[self.indexes[index]:self.indexes[index]+self.batch_size, 0].copy()
        if (self.out_channels == 1):
            mask = self.mask_dset[self.indexes[index]:self.indexes[index]+self.batch_size, :, :].copy()
            mask = np.expand_dims(mask, axis=3)
        else:
            mask = self.mask_dset[self.indexes[index]:self.indexes[index]+self.batch_size, :, :, :].copy()

        
        return data-0.5, mask
        
    def on_epoch_end(self):
        self.indexes = np.arange(0, self.dataset.attrs['elems'], self.batch_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

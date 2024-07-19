import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
import os

def convert_mitbih_to_hdf5(mitbih_path, output_hdf5):
    record_files = [f for f in os.listdir(mitbih_path) if f.endswith('.dat')]
    records = [os.path.splitext(f)[0] for f in record_files]
    
    with h5py.File(output_hdf5, 'w') as f:
        tracings_group = f.create_group('tracings')
        
        for record in records:
            record_path = os.path.join(mitbih_path, record)
            signals, fields = wfdb.rdsamp(record_path)
            annotations = wfdb.rdann(record_path, 'atr')
            tracings_group.create_dataset(record, data=signals)
             ann_group = tracings_group.create_group(record + '_ann')
            ann_group.create_dataset('sample', data=annotations.sample)
            ann_group.create_dataset('symbol', data=np.array(annotations.symbol, dtype='S'))
        
    print(f"Data converted to {output_hdf5}")
    mitbih_path = './mitbih_data'
output_hdf5 = './mitbih.h5'
convert_mitbih_to_hdf5(mitbih_path, output_hdf5)


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()

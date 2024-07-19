import wfdb
import h5py
import numpy as np
records = ['100', '101', '102', ...]
with h5py.File('mitbih.hdf5', 'w') as f:
    for record in records:
        # Load the record
        signal, fields = wfdb.rdsamp(record, pb_dir='mitdb')
        # Create a dataset in HDF5 file
        f.create_dataset(record, data=signal)
        # You might also want to store the annotations
        annotation = wfdb.rdann(record, 'atr', pb_dir='mitdb')
        f.create_dataset(f'{record}_annotations', data=annotation.sample)
        from tensorflow.keras.utils import Sequence
import numpy as np

class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples * (1 - val_split))
        train_seq = cls(path_to_hdf5, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, path_to_csv=None, batch_size=8, start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.records = list(self.f.keys())
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.records)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        batch_records = self.records[start:end]
        x = np.array([self.f[record][:] for record in batch_records])
        if self.y is None:
            return x
        else:
            y = np.array(self.y[start:end])
            return x, y

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()



            

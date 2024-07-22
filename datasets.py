import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split

annotation_to_class = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'E': 0,  # Non-Ectopic Beats
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular Ectopic Beats
    'V': 2, 'E': 2, 'F': 2,                  # Ventricular Ectopic Beats
    'F': 3,                                  # Fusion Beats
    'Q': 4                                   # Unknown Beats
}
def load_mitbih_data(data_dir):
    record_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    data = []
    labels = []

    for record_file in record_files:
        record_path = os.path.join(data_dir, record_file[:-4])
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        signal = record.p_signal
        raw_labels = annotation.symbol
        label = map_annotations_to_classes(raw_labels)
        
        if len(label) > 0:
            data.append(signal)
            labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)

 X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def map_annotations_to_classes(raw_labels):
    return [annotation_to_class[label] for label in raw_labels if label in annotation_to_class]


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

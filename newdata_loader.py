pip install wfdb numpy scipy 
import wfdb 
import numpy as np 
import os 
def load_mitbih_data(data_dir):
    record_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    data = []
    labels = []

    for record_file in record_files:
        record_path = os.path.join(data_dir, record_file[:-4])
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        signal = record.p_signal
        label = annotation.symbol
        
        data.append(signal)
        labels.append(label)
    
    return np.array(data), np.array(labels)

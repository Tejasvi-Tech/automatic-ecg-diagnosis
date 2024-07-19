import wfdb
import h5py
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


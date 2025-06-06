import os
import time
import torch
import torch.nn as nn
import pickle
import argparse
import numpy as np
from config.parser import YAMLParser
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

def generate_input_windows(labels, window_size, stride):
    """
    labels: numpy array shape of (T,)
    window_size: Size of window (step_num_in)
    stride: Stride of window
    """
    total_len = labels.shape[0]
    windows = []

    for i in range(0, total_len - window_size + 1, stride):
        window = labels[i : i + window_size]
        windows.append(window)

    return np.array(windows)


# Usage: python preprocess.py --config_path ./config/star.yaml --data_name SWaT
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/star.yaml')
    parser.add_argument('--data_name', type=str, default='SWaT', help='Data set to train.')
    parser.add_argument('--test', action='store_true', help='Test mode.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    args = parser.parse_args()

    configs = YAMLParser(args.config_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_args = getattr(configs, args.data_name, None)
    if data_args is None:
        raise KeyError(f'{args.data_name} not found in configs.')
    
    data_name = args.data_name
    step_num_in = data_args.step_num_in
    stride = getattr(configs, 'stride', None)


    print(time.strftime('%Y.%m.%d - %H:%M:%S'), f"Starting {data_name} ...")

    if data_name in ['SWaT']:
        # Read pickle file
        with open(f"./data/{data_name}/{data_name}.pk", 'rb') as f:
            loaded_dict = pickle.load(f)

        # Downsampling data
        x_trn = loaded_dict['x_trn'][::5]
        x_tst = loaded_dict['x_tst'][::5]
        lab_tst = loaded_dict['lab_tst'][::5]

        ignore_dims = getattr(data_args, "ignore_dims", [])
        if ignore_dims:
            print(f"❗ Ignoring dims: {ignore_dims}")
            x_trn = drop_ignore_dims(x_trn, ignore_dims)
            x_tst = drop_ignore_dims(x_tst, ignore_dims)

        scaler = StandardScaler() # MinMaxScaler()
        
        x_trn = scaler.fit_transform(x_trn) 
        x_tst = scaler.transform(x_tst)

        print("Shape of train:", x_trn.shape)
        print("Shape of test:", x_tst.shape)
        print("Shape of label:", lab_tst.shape)

        # original x_train, x_test
        x_trn_og = generate_input_windows(x_trn, step_num_in, stride)
        x_tst_og = generate_input_windows(x_tst, step_num_in, stride)
        np.save(f'./data/{data_name}/x_trn_{step_num_in}.npy', x_trn_og)
        np.save(f'./data/{data_name}/x_tst_{step_num_in}.npy', x_tst_og)
        print(f"✅ Saved: x_trn_{step_num_in}.npy, x_tst_{step_num_in}.npy")

        # Label
        label_windows = generate_input_windows(lab_tst, step_num_in, stride)
        # Save
        np.save(f'./data/{data_name}/labels_{step_num_in}.npy', label_windows)
        print("Shape of label_windows:", label_windows.shape)
        print(f"✅ Saved: labels_{step_num_in}.npy")

        # Finished SWaT
        print(time.strftime('%Y.%m.%d - %H:%M:%S'), f"Finished {data_name} ...")
    
    elif data_name in ['SMD']:
        # Read pickle file
        with open(f"./data/{data_name}/{data_name}.pk", 'rb') as f:
            loaded_dict = pickle.load(f)

        print(type(loaded_dict))              # Check Instance of loaded pickle data
        print(loaded_dict.keys())

        new_list = []
        for i in range(len(loaded_dict['f_name'])):
            f_name = loaded_dict['f_name'][i]
            # Using the addressed instances (TranAD)
            if f_name in ['machine-1-1.txt', 'machine-2-1.txt', 'machine-3-2.txt', 'machine-3-7.txt']: 
                new_list.append(i)
        for i in range(len(new_list)):
            print(i, loaded_dict['f_name'][new_list[i]])
            save_dir = f"./data/{data_name}/{data_name}_{i+1}"
            os.makedirs(save_dir, exist_ok=True)

            x_trn = loaded_dict['x_trn'][i]
            x_tst = loaded_dict['x_tst'][i]
            lab_tst = loaded_dict['lab_tst'][i]

            scaler = StandardScaler()

            x_trn = scaler.fit_transform(x_trn) 
            x_tst = scaler.transform(x_tst)

            print("Shape of train:", x_trn.shape)
            print("Shape of test:", x_tst.shape)
            print("Shape of label:", lab_tst.shape)


            # original x_train, x_test
            x_trn_og = generate_input_windows(x_trn, step_num_in, stride)
            x_tst_og = generate_input_windows(x_tst, step_num_in, stride)
            np.save(f'./data/{data_name}/{data_name}_{i+1}/x_trn_{step_num_in}.npy', x_trn_og)
            np.save(f'./data/{data_name}/{data_name}_{i+1}/x_tst_{step_num_in}.npy', x_tst_og)
            print(f"✅ Saved: x_trn_{step_num_in}.npy, x_tst_{step_num_in}.npy")

            # Label
            label_windows = generate_input_windows(lab_tst, step_num_in, stride)
            # Save
            np.save(f'./data/{data_name}/{data_name}_{i+1}/labels_{step_num_in}.npy', label_windows)
            print("Shape of label_windows:", label_windows.shape)
            print(f"✅ Saved: labels_{step_num_in}.npy")

        # Finished
        print(time.strftime('%Y.%m.%d - %H:%M:%S'), f"Finished {data_name} ...")
        
    
    elif data_name in ['SMAP', 'MSL']:
        # Read pickle file
        with open(f"./data/{data_name}/{data_name}.pk", 'rb') as f:
            loaded_dict = pickle.load(f)

        print(type(loaded_dict))              # Check Instance of loaded pickle data
        print(loaded_dict.keys())

        for i in range(len(loaded_dict['f_name'])):
            print(i, loaded_dict['f_name'][i])
            save_dir = f"./data/{data_name}/{data_name}_{i+1}"
            os.makedirs(save_dir, exist_ok=True)

            x_trn = loaded_dict['x_trn'][i]
            x_tst = loaded_dict['x_tst'][i]
            lab_tst = loaded_dict['lab_tst'][i]

            scaler = StandardScaler()

            x_trn = scaler.fit_transform(x_trn) 
            x_tst = scaler.transform(x_tst)

            print("Shape of train:", x_trn.shape)
            print("Shape of test:", x_tst.shape)
            print("Shape of label:", lab_tst.shape)


            # original x_train, x_test
            x_trn_og = generate_input_windows(x_trn, step_num_in, stride)
            x_tst_og = generate_input_windows(x_tst, step_num_in, stride)
            np.save(f'./data/{data_name}/{data_name}_{i+1}/x_trn_{step_num_in}.npy', x_trn_og)
            np.save(f'./data/{data_name}/{data_name}_{i+1}/x_tst_{step_num_in}.npy', x_tst_og)
            print(f"✅ Saved: x_trn_{step_num_in}.npy, x_tst_{step_num_in}.npy")

            # Label
            label_windows = generate_input_windows(lab_tst, step_num_in, stride)
            # Save
            np.save(f'./data/{data_name}/{data_name}_{i+1}/labels_{step_num_in}.npy', label_windows)
            print("Shape of label_windows:", label_windows.shape)
            print(f"✅ Saved: labels_{step_num_in}.npy")

        # Finished
        print(time.strftime('%Y.%m.%d - %H:%M:%S'), f"Finished {data_name} ...")

    else:
        print(f"{data_name} does not exist")

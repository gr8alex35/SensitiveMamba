import os
import torch
import argparse
import sensitive_hue
from sensitive_hue.model import *
import numpy as np
import torch.optim as optim
from config.parser import YAMLParser
from base.dataset import ADataset, split_dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

"""
Splitting Dataset
"""

def split_dataset(x, split_ratio=0.8):
    total_len = x.shape[0]
    train_len = int(total_len * split_ratio)

    train = x[:train_len]    
    val = x[train_len:]

    return train, val


class TSAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, x, label=None):
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.x[idx], self.label[idx]
        else:
            return self.x[idx]

"""
Getting Data Loaders
"""

def get_data_loaders(data_args, args, data_name, data_dir=None):
    step_num_in = data_args.step_num_in
    data_name = data_name
    batch_size = data_args.batch_size

    if data_dir != None:
        x_trn = np.load(f'{data_dir}/x_trn_{step_num_in}.npy')
        x_tst = np.load(f'{data_dir}/x_tst_{step_num_in}.npy')
        test_label = np.load(f'{data_dir}/labels_{step_num_in}.npy')    
    else:
        x_trn = np.load(f'./data/{data_name}/x_trn_{step_num_in}.npy')
        x_tst = np.load(f'./data/{data_name}/x_tst_{step_num_in}.npy')
        test_label = np.load(f'./data/{data_name}/labels_{step_num_in}.npy')

    train, val = split_dataset(x_trn, split_ratio=0.8)

    train_dataset = TSAD_Dataset(train)
    val_dataset = TSAD_Dataset(val)
    test_dataset = TSAD_Dataset(x_tst, test_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_loaders = [train_loader, val_loader, test_loader]

    return data_loaders


"""
train_single_entity
"""

def train_single_entity(args, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    data_loaders = get_data_loaders(data_args, args, data_name)
    # ------------------------ Trainer Setting ----------------------------
    model = SensitiveMamba(
        data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
        data_args.dim_hidden_fc, data_args.encode_layer_num, 0.2
    ).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    trainer = sensitive_hue.Trainer(
        model, optimizer, data_args.alpha, args.max_epoch, data_args.model_save_dir,
        scheduler, use_prob=True, model_save_suffix=f'{data_args.step_num_in}')
    
    if not only_test:
        trainer.train(data_loaders[0], data_loaders[1])
    
    ignore_dims = None 
    trainer.test(data_loaders[-1], ignore_dims, data_args.select_pos)

def train_multi_entity(args, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    start, end = data_args.range
    data_dir = data_args.data_dir

    ignore_dims = getattr(data_args, 'ignore_dims', dict())
    ignore_entities = getattr(data_args, 'ignore_entities', tuple())

    results = []
    for i in range(start, end + 1):
        if i in ignore_entities:
            continue
        data_args.data_dir = os.path.join(data_dir, f'{data_name}_{i}')
        data_loaders = get_data_loaders(data_args, args, data_name, data_args.data_dir)

        model = sensitive_hue.SensitiveMamba(
            data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
            data_args.dim_hidden_fc, data_args.encode_layer_num, 0.2
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
        trainer = sensitive_hue.Trainer(
            model, optimizer, data_args.alpha, args.max_epoch, data_args.model_save_dir,
            scheduler, use_prob=True, model_save_suffix=f'_{i}'
        )

        trainer.logger.info(f'entity {i}')

        if not only_test:
            trainer.train(data_loaders[0], data_loaders[1])
        cur_ignore_dim = ignore_dims[i] if i in ignore_dims else None
        ret = trainer.test(data_loaders[-1], cur_ignore_dim, data_args.select_pos)
        results.append(ret)

    results = np.concatenate(results, axis=1)
    trainer.show_metric_results(*results, prefix='Average')



# Usage: python main.py --config_path ./config/star.yaml --data_name SWaT
# Usage: CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --config_path ./config/star.yaml --data_name SWaT > nohup_1.txt &
# Usage: CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --config_path ./config/star.yaml --data_name SMD > nohup_2.txt &
# Usage: CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --config_path ./config/star.yaml --data_name MSL > nohup_3.txt &
# Usage: CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --config_path ./config/star.yaml --data_name SMAP > nohup_4.txt &
# Test:  python main.py --config_path ./config/star.yaml --data_name SWaT --test
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/star.yaml')
    parser.add_argument('--data_name', type=str, default='SWaT', help='Data set to train.')
    parser.add_argument('--override_step_num_in', type=int, default=None)
    parser.add_argument('--test', action='store_true', help='Test mode.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    args = parser.parse_args()

    configs = YAMLParser(args.config_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_args = getattr(configs, args.data_name, None)
    if data_args is None:
        raise KeyError(f'{args.data_name} not found in configs.')
    if hasattr(data_args, 'range'):
        train_multi_entity(configs, args.data_name, args.test)
    else:
        train_single_entity(configs, args.data_name, args.test)

import os
import logging
import numpy as np
from scipy.ndimage import convolve1d
from torch.utils import data
import torch

from utils import get_lds_kernel_window

print = logging.info


class PTBXLDataset(data.Dataset):
    """
    PTB-XL ECG Dataset for age prediction.
    Data is stored as numpy arrays with shape (N, timesteps, 12 leads)
    """
    def __init__(self, df, data_dir, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.split = split
        
        # Cache for loaded data
        self._data_cache = {}
        
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        
        # Load ECG data from numpy file
        file_path = os.path.join(self.data_dir, row['path'])
        sample_idx = int(row['index'])
        
        # Cache data file to avoid reloading
        if file_path not in self._data_cache:
            self._data_cache[file_path] = np.load(file_path)
        
        ecg_data = self._data_cache[file_path][sample_idx]  # Shape: (1000, 12)
        
        # Convert to tensor and reshape: (1000, 12) -> (12, 1000) for Conv1D
        ecg_tensor = torch.from_numpy(ecg_data.T).float()  # (12, 1000)
        
        label = np.asarray([row['age']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        return ecg_tensor, label, weight

    def _prepare_weights(self, reweight, max_target=100, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
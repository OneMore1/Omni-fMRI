import os
import glob
import numpy as np
from typing import Any, Callable, Dict, Optional, Set, Tuple
import torch
from torch.utils.data import Dataset
import random

class fMRIDataset(Dataset):
    def __init__(self, 
                 data_root, datasets, split_suffixes, crop_length=40, downstream=False):
        """
        初始化 Dataset
        """
        self.file_paths = []
        self.crop_length = crop_length
        self.downstream = downstream
        for dataset_name in datasets:
            for suffix in split_suffixes:
                folder_name = f"{dataset_name}_{suffix}"
                folder_path = os.path.join(data_root, folder_name)
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder not found: {folder_path}")
                    continue

                for root, dirs, files in os.walk(folder_path):
                    npz_files = glob.glob(os.path.join(root, "*.npz"))
                    if len(npz_files) > 1:
                        # sample_size = max(1, int(len(npz_files) * 0.5)) 
                        # npz_files = random.sample(npz_files, sample_size)
                        npz_files = sorted(npz_files)[:1]
                    self.file_paths.extend(npz_files)

        print(f"Dataset loaded. Total files found: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with np.load(file_path) as data_file:
                key = list(data_file.keys())[0]
                fmri_data = data_file[key] 
                fmri_data = fmri_data.astype(np.float32)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

        total_time_frames = fmri_data.shape[-1]
        if total_time_frames > self.crop_length:
            start_idx = np.random.randint(0, total_time_frames - self.crop_length + 1)
            end_idx = start_idx + self.crop_length
            cropped_data = fmri_data[..., start_idx:end_idx]
        else:
            cropped_data = fmri_data[..., :self.crop_length]

        data_tensor = torch.from_numpy(cropped_data)
        data_tensor = data_tensor.permute(3, 0, 1, 2)

        return data_tensor

'''
Author: ViolinSolo
Date: 2025-10-31 18:04:21
LastEditTime: 2025-10-31 20:19:13
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/test/test_dataloader.py
'''

import os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.fmri_adj_dataset import NpyMemmapDataset, collate_fn

def test_npy_memmap_dataset(paths_txt='test/data/paths.txt', labels_csv='test/data/labels.csv'):
    
    dataset = NpyMemmapDataset(
        paths_txt=paths_txt,
        labels_csv=labels_csv,
        path_col='npy_file',
        label_col='dx_cls_group',
        mmap_mode='r',
        transform=None,
        return_path=True,
        dtype=np.float32
    )
    
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        sample = dataset[i]
        data, label, path = sample['data'], sample['label'], sample['path']
        print(f"Sample {i}: data shape {data.shape}, label {label}, path {path}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        data_batch, label_batch, path_batch = batch['data'], batch['label'], batch['path']
        print(f"Batch data shape: {data_batch.shape}, label shape: {label_batch.shape}, path shape: {path_batch}")
        break  # Just test one batch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        data_batch, label_batch = batch['data'], batch['label']
        print(f"Batch data shape: {data_batch.shape}, label shape: {label_batch.shape}")
        break  # Just test one batch

if __name__ == "__main__":

    all_needed = [
        ('data/abide_adj/jepa/00_jepa_test_npy_paths.txt', 'data/abide_adj/jepa/00_jepa_test_matched_labels.csv'),
        ('data/abide_adj/jepa/00_jepa_train_npy_paths.txt', 'data/abide_adj/jepa/00_jepa_train_matched_labels.csv'),
        ('data/abide_adj/jepa/00_jepa_val_npy_paths.txt', 'data/abide_adj/jepa/00_jepa_val_matched_labels.csv'),

        ('data/abide_adj/lm/00_lm_test_npy_paths.txt', 'data/abide_adj/lm/00_lm_test_matched_labels.csv'),
        ('data/abide_adj/lm/00_lm_train_npy_paths.txt', 'data/abide_adj/lm/00_lm_train_matched_labels.csv'),
        ('data/abide_adj/lm/00_lm_val_npy_paths.txt', 'data/abide_adj/lm/00_lm_val_matched_labels.csv'),

        ('data/abide_adj/mass/00_mass_test_npy_paths.txt', 'data/abide_adj/mass/00_mass_test_matched_labels.csv'),
        ('data/abide_adj/mass/00_mass_train_npy_paths.txt', 'data/abide_adj/mass/00_mass_train_matched_labels.csv'),
        ('data/abide_adj/mass/00_mass_val_npy_paths.txt', 'data/abide_adj/mass/00_mass_val_matched_labels.csv'),
    ]
    
    for paths_txt, labels_csv in all_needed:
        print("--------------------------------------------------")
        print(f"\nTesting NpyMemmapDataset with paths: {paths_txt} and labels: {labels_csv}")
        test_npy_memmap_dataset(paths_txt, labels_csv)
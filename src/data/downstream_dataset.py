import os
import glob
import re 
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Literal
import torch.nn.functional as F
from .pretrain_dataset import fMRIDataset
import io  
import nibabel as nib

class fMRITaskDataset(fMRIDataset):

    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        split_suffixes: List[str],
        crop_length: int,
        label_csv_path: str,
        task_type: Literal['classification', 'regression'] = 'classification',
        downstream=True,
    ):

        super().__init__(data_root, datasets, split_suffixes, crop_length, downstream)
        
        self.task_type = task_type
        self.labels_map = self._load_and_process_labels(label_csv_path)

        initial_file_count = len(self.file_paths)
        self.file_paths = [
            path for path in self.file_paths 
            if self._extract_subject_id(path) in self.labels_map
        ]
        
        if len(self.file_paths) < initial_file_count:
            print(f"Warning: Dropped {initial_file_count - len(self.file_paths)} files due to missing labels in CSV.")
        
        print(f"Task Dataset ready for {self.task_type}. Usable files: {len(self.file_paths)}")


    def _extract_subject_id(self, file_path: str) -> str:

            folder_name = os.path.basename(os.path.dirname(file_path))
            match = re.search(r'(\d{7})', folder_name)

            # match = re.search(r'(\d{6})', os.path.basename(file_path))
            
            if match:
                subject_id_with_zeros = match.group(1)
                subject_id = subject_id_with_zeros.lstrip('0') # 去除前缀0
                
                return subject_id
                
            return "" 


    def _load_and_process_labels(self, csv_path: str) -> dict:

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label CSV file not found at: {csv_path}")
            
        print(f"Loading labels from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df['Subject'] = df['Subject'].astype(str)
        df.dropna(subset=['Subject'], inplace=True) 

        labels_map = {}
        
        if self.task_type == 'classification':
            label_col = None
            if 'DX_GROUP' in df.columns:
                label_col = 'DX_GROUP'
            elif 'gender' in df.columns:
                label_col = 'gender'
            elif 'age_group' in df.columns: 
                label_col = 'age_group'
            
            if label_col is None:
                raise ValueError("CSV must contain 'sex', 'gender' or 'age_group' column for classification.")

            print(f"Using column '{label_col}' as label.")

            sex_mapping = {'F': 0, 'M': 1, 'f': 0, 'm': 1}
            
            if df[label_col].dtype == object and df[label_col].astype(str).iloc[0].upper() in ['F', 'M']:
                print(f"Encoding {label_col} (F/M) to Integers (0/1)...")
                df = df[df[label_col].isin(sex_mapping.keys())]
                df[label_col] = df[label_col].map(sex_mapping)
            else:
                df[label_col] = pd.to_numeric(df[label_col], errors='coerce').astype(int)

            for _, row in df.iterrows():
                subject_id = row['Subject']
                labels_map[subject_id] = torch.tensor(row[label_col], dtype=torch.long)

        elif self.task_type == 'regression':
            label_col = 'age'
            if label_col not in df.columns:
                 raise ValueError(f"Regression task requires '{label_col}' column.")
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
            df.dropna(subset=[label_col], inplace=True)
            
            for _, row in df.iterrows():
                subject_id = row['Subject']
                labels_map[subject_id] = torch.tensor(row[label_col], dtype=torch.float32).view(1)

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        print(f"Successfully loaded {len(labels_map)} subjects' labels.")
        return labels_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        retries = 0
        max_retries = 100 
        while retries < max_retries:
            try:
                data_tensor = super().__getitem__(idx)
                
                if data_tensor is None:
                    raise ValueError(f"Failed to load data at index {idx} (super returned None)")

                file_path = self.file_paths[idx]
                
                subject_id = self._extract_subject_id(file_path)

                if subject_id in self.labels_map:
                    label_tensor = self.labels_map[subject_id]
                    return data_tensor, label_tensor
                else:
                    raise KeyError(f"Label not found for subject ID: {subject_id}")

            except Exception as e:
                
                idx = np.random.randint(0, len(self))
                retries += 1
        
        raise RuntimeError(f"Failed to load any valid data after {max_retries} retries.")
            
        return data_tensor, label_tensor


class fMRITaskDataset1(fMRIDataset):

    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        split_suffixes: List[str],
        crop_length: int,
        label_csv_path: str,
        task_type: Literal['classification', 'regression'] = 'classification',
        downstream: bool = True,
        subject_list_txt: str = None, 
        samples_per_subject: int = 1 
    ):
        super().__init__(data_root, datasets, split_suffixes, crop_length, downstream)
        
        self.task_type = task_type

        self.labels_map = self._load_and_process_labels(label_csv_path)

        if subject_list_txt and os.path.exists(subject_list_txt):
            print(f"Dataset Mode: TXT List ({subject_list_txt})")
            self.file_paths = self._load_files_from_txt(subject_list_txt, samples_per_subject)
        else:
            print("Dataset Mode: Automatic Scan (Standard)")
            initial_file_count = len(self.file_paths)
            self.file_paths = [
                path for path in self.file_paths 
                if self._extract_subject_id(path) in self.labels_map
            ]
            if len(self.file_paths) < initial_file_count:
                print(f"Warning: Dropped {initial_file_count - len(self.file_paths)} files due to missing labels.")
        
        print(f"Task Dataset ready for {self.task_type}. Total usable files: {len(self.file_paths)}")

    def _load_files_from_txt(self, txt_path: str, limit_per_subject: int) -> List[str]:

        valid_files = []
        
        with open(txt_path, 'r') as f:
            subject_folders = [line.strip() for line in f if line.strip()]

        print(f"Found {len(subject_folders)} subject folders in txt.")

        for folder_path in subject_folders:
            if not os.path.exists(folder_path):
                continue
                
            subject_id = self._extract_subject_id(folder_path)

            if subject_id not in self.labels_map:
                continue
            
            subject_files = sorted(glob.glob(os.path.join(folder_path, '*.npy'))) 
            
            if not subject_files:
                 subject_files = sorted([
                     os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))
                 ])

            if limit_per_subject is not None and limit_per_subject > 0:
                subject_files = subject_files[:limit_per_subject]
            
            valid_files.extend(subject_files)

        return valid_files

    def _extract_subject_id(self, path: str) -> str:

        regex = r'(\d{6})'
        
        name = os.path.basename(path)
        match = re.search(regex, name)
        # match = re.search(r'nest(\d+)', name)
        
        if not match:
            parent_name = os.path.basename(os.path.dirname(path))
            match = re.search(regex, parent_name)
            # match = re.search(r'nest(\d+)', parent_name)

        if match:
            subject_id_with_zeros = match.group(1)
            subject_id = subject_id_with_zeros.lstrip('0')
            return subject_id
        return "" 

    def _load_and_process_labels(self, csv_path: str) -> dict:

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label CSV file not found at: {csv_path}")
            
        df = pd.read_csv(csv_path)

        df['Subject'] = df['Subject'].astype(str)
        df.dropna(subset=['Subject'], inplace=True) 

        labels_map = {}
        
        if self.task_type == 'classification':
            label_col = None
            for col in ['Gender']: 
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                raise ValueError("Could not find a valid label column (e.g., 'Gender') in CSV.")

            sex_mapping = {'F': 0, 'M': 1, 'f': 0, 'm': 1}
            

            temp_check = df[label_col].dropna()
            if not temp_check.empty and temp_check.dtype == object and str(temp_check.iloc[0]).upper() in ['F', 'M']:
                df = df[df[label_col].isin(sex_mapping.keys())]
                df[label_col] = df[label_col].map(sex_mapping)
            else:
                df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
                
                original_count = len(df)
                
                df.dropna(subset=[label_col], inplace=True)
                
                dropped_count = original_count - len(df)
                if dropped_count > 0:
                    print(f"Info: Skipped {dropped_count} subjects due to missing/invalid '{label_col}' labels.")

                df[label_col] = df[label_col].astype(int) 
            
            for _, row in df.iterrows():
                subj_str = row['Subject']
                if subj_str.isdigit():
                    subj_str = subj_str.lstrip('0')
                
                labels_map[subj_str] = torch.tensor(row[label_col], dtype=torch.long)

        elif self.task_type == 'regression':
            label_col = 'age'
            if label_col not in df.columns:
                 raise ValueError(f"Regression task requires '{label_col}' column.")
            
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
            df.dropna(subset=[label_col], inplace=True)
            
            for _, row in df.iterrows():
                subj_str = row['Subject']
                if subj_str.isdigit():
                    subj_str = subj_str.lstrip('0')
                labels_map[subj_str] = torch.tensor(row[label_col], dtype=torch.float32).view(1)

        return labels_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        retries = 0
        max_retries = 30
        
        while retries < max_retries:
            try:
                data_tensor = super().__getitem__(idx)
                
                if data_tensor is None:
                     raise ValueError("Super class returned None")

                file_path = self.file_paths[idx]
                subject_id = self._extract_subject_id(file_path)
                
                if subject_id in self.labels_map:
                    label_tensor = self.labels_map[subject_id]
                    return data_tensor, label_tensor
                else:
                    raise KeyError(f"Label not found for subject ID: {subject_id}")

            except Exception: 
                idx = np.random.randint(0, len(self))
                retries += 1
        
        raise RuntimeError(f"Failed to load data after {max_retries} retries.")
    

class EmoFMRIDataset(Dataset):
    def __init__(self, txt_file, csv_dir, crop_length=40, normalize=True):

        self.crop_length = crop_length
        self.csv_dir = csv_dir
        self.normalize = normalize
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Txt file not found: {txt_file}")
            
        with open(txt_file, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        self.labels_cache = {}
        self._preload_all_csvs()

    def _preload_all_csvs(self):

        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv') and 'sub' in f]

        for filename in csv_files:
            match = re.search(r'(sub-[a-zA-Z0-9]+)', filename)
            if match:
                sub_key = match.group(1)
            else:
                sub_key = os.path.splitext(filename)[0]

            csv_path = os.path.join(self.csv_dir, filename)
            
            df = pd.read_csv(csv_path, header=None)
            self.labels_cache[sub_key] = df.iloc[:, 0].values


        print("标签加载完成。")

    def _apply_global_zscore(self, data):

        mask = np.abs(data) > 1e-2
        
        if mask.sum() == 0:
            return data
        
        brain_voxels = data[mask]
        mean = brain_voxels.mean()
        std = brain_voxels.std()

        if std < 1e-8:
            std = 1.0
            
        data_normalized = np.zeros_like(data)
        data_normalized[mask] = (brain_voxels - mean) / std
        data_normalized = np.maximum(data_normalized, 0.0)
        
        return data_normalized

    def _parse_filename(self, file_path):

        filename = os.path.basename(file_path)

        match_sub = re.search(r'(sub-[a-zA-Z0-9]+)', filename)

        sub_id = match_sub.group(1)
        
        match_point = re.search(r'point_(\d+)', filename)
        
        if match_point:
            idx = int(match_point.group(1)) - 1
        else:
            idx = 0
            
        return sub_id, idx

    def load_data(self, file_path):
        if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):

            img = nib.load(file_path)
            data = img.get_fdata(dtype=np.float32)
            if np.isnan(data).any():
                data = np.nan_to_num(data)
            return data


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        sub_id, row_index = self._parse_filename(path)
        
        target_key = sub_id
        
        sub_labels = self.labels_cache[target_key]
            
        label_value = sub_labels[row_index]
        fmri_data = self.load_data(path) 

        if self.normalize:
            fmri_data = self._apply_global_zscore(fmri_data)

        total_time_frames = fmri_data.shape[-1]
        
        if self.crop_length > 0:
            if total_time_frames > self.crop_length:
                start_idx = np.random.randint(0, total_time_frames - self.crop_length + 1)
                cropped_data = fmri_data[..., start_idx : start_idx + self.crop_length]
            else:
                if total_time_frames < self.crop_length:
                     repeats = (self.crop_length // total_time_frames) + 1
                     cropped_data = np.repeat(fmri_data, repeats, axis=-1)[..., :self.crop_length]
                else:
                    cropped_data = fmri_data
        else:
            cropped_data = fmri_data

        data_tensor = torch.from_numpy(cropped_data.copy())
        data_tensor = data_tensor.permute(3, 0, 1, 2) # T, X, Y, Z
        
        label_tensor = torch.tensor(label_value, dtype=torch.float32)

        point_id_tensor = torch.tensor(row_index, dtype=torch.long)
        
        return data_tensor, label_tensor


import zstandard as zstd

class HCPtaskDataset(Dataset):
    def __init__(self, txt_path, time_length=40, padding_mode='zeros', transform=None):

        self.time_length = time_length
        self.transform = transform
        self.file_paths = self._load_txt(txt_path)

        if padding_mode not in ['zeros', 'repeat']:
            raise ValueError(f"padding_mode must be 'zeros' or 'repeat', got '{padding_mode}'")
        self.padding_mode = padding_mode

        self.label_to_id, self.id_to_label = self._build_label_mapping()
        self.num_classes = len(self.label_to_id)

    def _load_txt(self, txt_path):
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")
        
        with open(txt_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths

    def _extract_label_from_path(self, path):
        filename = os.path.basename(path)
        name_no_ext = filename.split('.')[0] 
        parts = name_no_ext.split('_')
        
        start_idx = parts.index('tfMRI') + 1
        end_idx = -1 
        raw_label_parts = parts[start_idx:end_idx]
        clean_parts = [p for p in raw_label_parts if p not in ('LR', 'RL')]
        label = "_".join(clean_parts)

        return label


    def _build_label_mapping(self):
        unique_labels = set()
        for path in self.file_paths:
            label = self._extract_label_from_path(path)
            unique_labels.add(label)
        
        sorted_labels = sorted(list(unique_labels))
        label_to_id = {label: i for i, label in enumerate(sorted_labels)}
        id_to_label = {i: label for i, label in enumerate(sorted_labels)}
        return label_to_id, id_to_label

    def _load_zst_data(self, file_path):
        dctx = zstd.ZstdDecompressor()
        try:
            with open(file_path, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    content = reader.read()
            bio = io.BytesIO(content)
            data = np.load(bio)
            return data
        except Exception as e:
            raise e

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        try:
            data_np = self._load_zst_data(file_path)
        except Exception:
            data_np = np.zeros((96, 96, 96, self.time_length), dtype=np.float32)

        if data_np.ndim == 4:
            current_t = data_np.shape[-1]
            target_t = self.time_length

            if current_t < target_t:

                pad_width = target_t - current_t

                pad_config = ((0,0), (0,0), (0,0), (0, pad_width))
                
                if self.padding_mode == 'repeat':
                    data_np = np.pad(data_np, pad_config, mode='wrap')
                else: # 'zeros'
                    data_np = np.pad(data_np, pad_config, mode='constant', constant_values=0)
                
            elif current_t > target_t:
                data_np = data_np[..., :target_t]
        else:
            print(f"Warning: {file_path} shape abnormal: {data_np.shape}")
            data_np = np.zeros((96, 96, 96, self.time_length), dtype=np.float32)

        data_tensor = torch.from_numpy(data_np).float()

        label_str = self._extract_label_from_path(file_path)
        label_idx = self.label_to_id[label_str]
        
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            data_tensor = self.transform(data_tensor)
        data_tensor = data_tensor.permute(3, 0, 1, 2)

        return data_tensor, label_tensor
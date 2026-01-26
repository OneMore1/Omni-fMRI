import os
import re
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    def __init__(self, txt_file, crop_length=40, class_mapping={'cn': 0, 'ad': 1}, samples_per_subject=4, 
                 label_source='directory', csv_file=None):
        self.crop_length = crop_length
        self.class_mapping = class_mapping if class_mapping else {}
        self.samples_per_subject = samples_per_subject
        self.label_source = label_source
        
        if self.label_source == 'csv':
            self.csv_file = csv_file
            self._load_csv_labels() 

        self.subject_map = {} 
        self.epoch_file_paths = [] 

        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Txt file not found: {txt_file}")
            
        with open(txt_file, 'r') as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        for file_path in all_lines:
            sub_id = self.extract_subject_id(file_path)
            if sub_id:
                if sub_id not in self.subject_map:
                    self.subject_map[sub_id] = []
                self.subject_map[sub_id].append(file_path)
        
        print(f"Dataset initialized. Label source: {self.label_source}")
        print(f"Total subjects found in TXT: {len(self.subject_map)}")
        
        if self.label_source == 'csv':
            matched_count = sum(1 for sid in self.subject_map.keys() if sid in self.csv_label_map)
            print(f"Subjects matched in CSV: {matched_count} / {len(self.subject_map)}")

        self.shuffle_epoch()

    def _load_csv_labels(self):
        print(f"Loading labels from CSV: {self.csv_file}")
        try:
            df = pd.read_csv(self.csv_file)
            
            label_col = 'Gender' 
            
            df['Subject'] = df['Subject'].astype(str).str.strip()
            
            self.csv_label_map = df.set_index('Subject')[label_col].to_dict()
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise e

    def extract_subject_id(self, file_path):

        filename = os.path.basename(file_path)
        match = re.search(r'sub-([^_]+)_', filename)
        
        if match:
            return match.group(1) 
        else:
            return None

    def shuffle_epoch(self):
        self.epoch_file_paths = []
        for sub_id, files in self.subject_map.items():
            if len(files) >= self.samples_per_subject:
                selected_files = np.random.choice(files, self.samples_per_subject, replace=False)
            else:
                selected_files = np.random.choice(files, self.samples_per_subject, replace=True)
            self.epoch_file_paths.extend(selected_files)

    def __len__(self):
        return len(self.epoch_file_paths)

    def get_label(self, file_path):
        if self.label_source == 'csv':
            sub_id = self.extract_subject_id(file_path)

            if sub_id not in self.csv_label_map:
                raise ValueError(f"Subject ID [{sub_id}] extracted from file not found in CSV map keys.")
                
            raw_val = self.csv_label_map[sub_id]
            
            try:
                return int(raw_val)
            except ValueError:
                if raw_val in self.class_mapping:
                    return self.class_mapping[raw_val]
                else:
                    raise ValueError(f"Label '{raw_val}' (Subject: {sub_id}) is not int and not in class_mapping.")

        elif self.label_source == 'directory':
            raw_label_name = os.path.basename(os.path.dirname(file_path))
            if raw_label_name in self.class_mapping:
                return self.class_mapping[raw_label_name]
            else:
                raise ValueError(f"Directory label '{raw_label_name}' not found in class_mapping.")

    def load_data(self, file_path):
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            img = nib.load(file_path)
            data = img.get_fdata()
            return data.astype(np.float32)
        elif file_path.endswith('.npz'):
            with np.load(file_path, allow_pickle=True) as data_file:
                if 'data' in data_file:
                    data = data_file['data']
                else:
                    key = list(data_file.keys())[0]
                    data = data_file[key]
                return data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported file extension: {file_path}")

    def __getitem__(self, idx):
        file_path = self.epoch_file_paths[idx]
        
        label = self.get_label(file_path)

        try:
            fmri_data = self.load_data(file_path)
        except Exception as e:
            print(f"Error loading data: {file_path}")
            raise e

        total_time_frames = fmri_data.shape[-1]
        if total_time_frames > self.crop_length:
            start_idx = np.random.randint(0, total_time_frames - self.crop_length + 1)
            cropped_data = fmri_data[..., start_idx : start_idx + self.crop_length]
        else:
            if total_time_frames < self.crop_length:
                repeats = (self.crop_length // total_time_frames) + 1
                cropped_data = np.repeat(fmri_data, repeats, axis=-1)[..., :self.crop_length]
            else:
                cropped_data = fmri_data

        data_tensor = torch.from_numpy(cropped_data.copy())
        data_tensor = data_tensor.permute(3, 0, 1, 2)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return data_tensor, label_tensor

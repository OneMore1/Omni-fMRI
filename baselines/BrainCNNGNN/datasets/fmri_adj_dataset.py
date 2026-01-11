'''
Author: ViolinSolo
Date: 2025-10-31 17:22:27
LastEditTime: 2025-10-31 18:02:25
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/datasets/fmri_adj_dataset.py
'''

from typing import Optional, Callable, List, Dict, Any
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyMemmapDataset(Dataset):
    """
    Dataset of .npy files (one file path per line in a text file) with labels provided in a CSV.

    Parameters
    ----------
    paths_txt : str
        Path to a text file where each line is a path to a .npy file.
    labels_csv : str
        Path to a CSV file containing labels. By default expected columns are:
            - 'path' : path or filename that identifies the sample (can be full path, basename, or a suffix)
            - 'label': integer or any object representing the label
        You can change the column names with path_col and label_col.
    path_col : str
        Column name in labels_csv that contains an identifier for the .npy file (default 'path').
    label_col : str
        Column name in labels_csv that contains the label (default 'label').
    mmap_mode : Optional[str]
        Mode passed to np.load for memmap. Typical choice: 'r' (read-only). Default 'r'.
    transform : Optional[Callable]
        Optional callable applied to the loaded numpy array. Should accept a numpy array and return
        a tensor or array.
    return_path : bool
        If True, __getitem__ returns (data, label, path). Otherwise returns (data, label).
    dtype : Optional[np.dtype]
        If provided, cast the loaded array to this dtype before converting to torch.Tensor.
    """

    def __init__(
        self,
        paths_txt: str,
        labels_csv: str,
        path_col: str = "path",
        label_col: str = "label",
        mmap_mode: Optional[str] = "r",
        transform: Optional[Callable] = None,
        return_path: bool = False,
        dtype: Optional[np.dtype] = None,
    ):
        self.paths_txt = paths_txt
        self.labels_csv = labels_csv
        self.path_col = path_col
        self.label_col = label_col
        self.mmap_mode = mmap_mode
        self.transform = transform
        self.return_path = return_path
        self.dtype = dtype

        # read list of file paths from text file
        with open(self.paths_txt, "r", encoding="utf-8") as f:
            raw_paths = [ln.strip() for ln in f if ln.strip()]
        # normalize to absolute paths if possible (keeps comparability)
        self.file_paths: List[str] = [os.path.abspath(p) for p in raw_paths]

        # read labels csv
        df = pd.read_csv(self.labels_csv)
        if self.path_col not in df.columns or self.label_col not in df.columns:
            raise ValueError(
                f"labels_csv must contain columns '{self.path_col}' and '{self.label_col}'. "
                f"Found columns: {list(df.columns)}"
            )

        # Build mapping: try to match by absolute path first, then fallback to basename, then suffix match
        # Create helper maps
        self._label_map_fullpath: Dict[str, Any] = {}
        self._label_map_basename: Dict[str, Any] = {}
        self._label_map_suffix: List[tuple] = []

        for _, row in df.iterrows():
            key = str(row[self.path_col])
            label = row[self.label_col]
            # store as-is too
            absk = os.path.abspath(key) if os.path.isabs(key) or os.path.exists(key) else key
            self._label_map_fullpath[absk] = label
            basename = os.path.basename(key)
            self._label_map_basename[basename] = label
            # for suffix matching (e.g., relative paths), keep the original key
            self._label_map_suffix.append((str(key), label))

        # Assign labels for all file_paths
        self.labels: List[Any] = []
        missing = []
        for p in self.file_paths:
            lab = self._find_label_for_path(p)
            if lab is None:
                missing.append(p)
                self.labels.append(None)
            else:
                self.labels.append(lab)

        if missing:
            # If you prefer to raise, replace this with an exception.
            raise ValueError(
                "Some paths from paths_txt could not be matched to labels in labels_csv. "
                f"Missing {len(missing)} examples. Example missing paths: {missing[:5]}"
            )

        # memmap cache: lazy open per path -> stores np.memmap or None
        # We keep this as dict so each worker process can open its own memmaps lazily.
        self._memmaps: Dict[str, Optional[np.memmap]] = {p: None for p in self.file_paths}


        self.n_classes = self.get_num_classes()
        print(f"NpyMemmapDataset initialized with {len(self.file_paths)} samples, {self.n_classes} classes.")

    def get_num_classes(self) -> int:
        """Get number of unique classes in the dataset."""
        unique_labels = set(lab for lab in self.labels if lab is not None)
        return len(unique_labels)

    def _find_label_for_path(self, absolute_path: str):
        # 1) exact absolute match
        if absolute_path in self._label_map_fullpath:
            return self._label_map_fullpath[absolute_path]
        # 2) basename match
        b = os.path.basename(absolute_path)
        if b in self._label_map_basename:
            return self._label_map_basename[b]
        # 3) suffix match: find first label where key is a suffix of absolute_path
        for key, lab in self._label_map_suffix:
            if absolute_path.endswith(key) or os.path.basename(absolute_path).endswith(key):
                return lab
        return None

    def __len__(self):
        return len(self.file_paths)

    def _open_memmap(self, path: str) -> np.memmap:
        """Open (and cache) a memmap for path if not already opened."""
        mm = self._memmaps.get(path)
        if mm is None:
            # use numpy.load with mmap_mode to get a memmap-backed object
            arr = np.load(path, mmap_mode=self.mmap_mode)
            # np.load returns numpy.memmap object when mmap_mode is not None and file is .npy
            # but sometimes it returns ndarray view; cast to memmap-like object by using arr
            # We'll store arr (which is either memmap or ndarray).
            self._memmaps[path] = arr
            return arr
        return mm

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        label = self.labels[idx]

        arr = self._open_memmap(path)

        shape = arr.shape
        if len(shape) == 2:
            # expand the dims into (1, N, N)
            arr = np.expand_dims(arr, axis=0)

        # Optionally cast dtype (keeps memmap semantics if possible)
        if self.dtype is not None and arr.dtype != self.dtype:
            # Note: casting will create a copy
            arr = arr.astype(self.dtype)

        # Optionally apply transform (user can convert to torch inside transform)
        if self.transform is not None:
            data = self.transform(arr)
            # If transform returns numpy array, convert to torch
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
        else:
            # default conversion: numpy -> torch tensor
            # If arr is memmap, torch.from_numpy will create a tensor that views the memmap buffer.
            data = torch.from_numpy(np.asarray(arr))

        # One item
        sample = {'data': data, 'label': label}

        if self.return_path:
            sample['path'] = path
            return sample
        else:
            return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to handle batches of samples."""
    data_list = [item['data'] for item in batch]
    label_list = [item['label'] for item in batch]

    data_batch = torch.stack(data_list, dim=0)
    label_batch = torch.tensor(label_list)

    return {
        'data': data_batch, 
        'label': label_batch
    }

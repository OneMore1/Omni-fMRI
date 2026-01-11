'''
Author: ViolinSolo
Date: 2025-11-05 16:38:59
LastEditTime: 2025-11-27 12:37:00
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/datasets/fmri_graph_dataset.py
'''

from typing import Optional, Callable, List, Dict, Any
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
# from torch.utils.data import Dataset
from pathlib import Path


class NPYGraphDataset(Dataset):
    """
    PyG Dataset that reads .npy graph files produced by this script and returns torch_geometric.data.Data objects.

    Args:
        root (str or Path): directory containing .npy files (or a parent directory); this class will look for any *.npy files.
        transform, pre_transform: same as PyG Dataset API (passed to super)
        use_adj_as_x (bool): 如果 True，则 Data.x 会被设置为 adjacency 矩阵（R, R）；否则使用保存的 node_features (R, F)
        label_map (dict): 可选， mapping from filename -> label (int) 用于设置 data.y
    """

    def __init__(
        self, 
        root: str or Path,
        paths_txt: str, 
        labels_csv: str,
        path_col: str = "path",
        label_col: str = "label",
        transform: Optional[Callable] = None,
        dtype: Optional[np.dtype] = None,
        pre_transform: Optional[Callable] = None,
        use_adj_as_x=True,
    ):
        super().__init__(root, transform, pre_transform)
        self.root = Path(root)
        self.paths_txt = paths_txt
        self.labels_csv = labels_csv
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.dtype = dtype
        
        # read list of file paths from text file
        with open(self.paths_txt, "r", encoding="utf-8") as f:
            raw_paths = [ln.strip() for ln in f if ln.strip()]
        # load labels
        df = pd.read_csv(self.labels_csv)
        if self.path_col not in df.columns or self.label_col not in df.columns:
            raise ValueError(
                f"labels_csv must contain columns '{self.path_col}' and '{self.label_col}'. "
                f"Found columns: {list(df.columns)}"
            )
        
        self.data_info = []
        for fpath in raw_paths:
            fname = os.path.basename(fpath)
            if '_corr_graph.npy' in fname:
                label = df.loc[df[self.path_col] == fname, self.label_col].values[0]
                corr_fpth = fpath
                pcorr_fpth = fpath.replace('_corr_graph.npy', '_pcorr_graph.npy')
                self.data_info.append( (corr_fpth, pcorr_fpth, label) )
        
        # self.files = sorted(list(self.root.glob('*_graph.npy')) + [p for p in self.root.glob('*.npy') if not p.name.endswith('_graph.npy')])
        # filter out any non-graph npy? We assume the directory contains only our generated files or raw ones
        self.use_adj_as_x = use_adj_as_x

    def len(self):
    # def __len__(self):
        return len(self.data_info)

    def get(self, idx):
    # def __getitem__(self, idx):
        corr_fpath, pcorr_fpath, label = self.data_info[idx]
        data_dict_corr = np.load(str(corr_fpath), allow_pickle=True).item()
        adj = data_dict_corr.get('adj').astype(self.dtype)
        node_feats = data_dict_corr.get('node_features').astype(self.dtype)
        data_dict_pcorr = np.load(str(pcorr_fpath), allow_pickle=True).item()
        edge_index_np = data_dict_pcorr.get('edge_index').astype(np.int64)
        edge_attr_np = data_dict_pcorr.get('edge_attr').astype(self.dtype)
        num_nodes = adj.shape[0]
        pseudo = np.diag(np.ones(num_nodes)).astype(self.dtype)  # PE

        # build tensors
        if self.use_adj_as_x:
            x = torch.from_numpy(adj)  # (R, R)
        else:
            x = torch.from_numpy(node_feats)
        if edge_index_np is None or edge_attr_np is None:
            # fallback: build from adj
            edge_index, edge_attr = adj_to_edge_index_and_attr(adj)
        else:
            edge_index = torch.from_numpy(edge_index_np)
            edge_attr = torch.from_numpy(edge_attr_np)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=torch.from_numpy(pseudo))
        # also attach full adjacency if needed
        data.adj = torch.from_numpy(adj)

        # # try to set y if available in label_map
        # basename = fpath.stem
        # label = self.label_map.get(basename)
        if label is not None:
            if isinstance(label, float):  # regression
                data.y = torch.tensor([label], dtype=torch.float32)
            else:  # classification
                data.y = torch.tensor(label, dtype=torch.long)
        else:
            data.y = None

        # # attach metadata
        # data.meta = data_dict.get('meta', {})
        return data


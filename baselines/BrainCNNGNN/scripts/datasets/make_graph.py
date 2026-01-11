'''
Author: ViolinSolo
Date: 2025-11-05 11:45:04
LastEditTime: 2025-11-05 18:32:44
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/scripts/datasets/make_graph.py
'''
"""
convert_fmri_to_graphs.py

将 time x ROIs 的 fMRI (.npy) 转换为用于 PyG( PyTorch Geometric ) 的图数据结构（**不保存为 .pt**，仍然以 .npy 保存原始图信息），并提供一个加载这些 .npy 文件并返回 `torch_geometric.data.Data` 对象的 Dataset 类。

功能要点：
- 支持 Pearson correlation（默认）与 partial correlation（基于逆协方差）作为边权重计算方法
- 支持稀疏化：abs_thresh / top_k / top_percent
- 生成并保存以下字段到 .npy（字典）：
    - 'adj' : 完整 R×R 邻接矩阵（float32）
    - 'node_features' : R×F 节点特征（float32），mode 可选 stats/timeseries/pca
    - 'edge_index' : 2×E (int64) （按 PyG 格式）
    - 'edge_attr' : E (float32) 对应 edge_index 的权重
    - 'meta' : 保存生成参数与原始 shape 等信息

- 提供 `NPYGraphDataset`（继承自 `torch_geometric.data.Dataset`）用于在训练时直接返回 `torch_geometric.data.Data` 对象。Data 中包含：
    - data.x -> 节点特征（默认）或在初始化时可设置为把 adjacency 作为 x
    - data.edge_index, data.edge_attr
    - data.adj -> 完整邻接矩阵（如果你需要在模型里直接访问）
    - data.y -> placeholder（默认为 None，用户在训练环节可提供 label 映射）

注意：
- 本脚本**不把数据保存为 .pt**，只保存 .npy（满足你的要求）
- 需要安装：numpy, scikit-learn (用于 PCA 可选), torch, torch_geometric

用法示例：
    python convert_fmri_to_graphs.py --input_dir ./fmri_npy --output_dir ./graphs_npy --edge_method correlation --top_percent 10 --node_mode timeseries

在代码中也同时包含了 `process_file` 函数和批量处理函数，和返回 PyG Data 的 Dataset 类。
"""

import os
from pathlib import Path
import argparse
import numpy as np
from sklearn.decomposition import PCA

# PyTorch / PyG
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def load_fmri(path: Path, n_rois: int=None):
    x = np.load(str(path))
    if x.ndim != 2:
        raise ValueError(f"Expect 2D numpy array, got shape {x.shape} for {path}")
    t, r = x.shape
    if n_rois is None:
        # heuristic: if time < roi, assume input was (roi, time)
        if t < r:
            x = x.T
    else:
        if r != n_rois:
            if t == n_rois:
                x = x.T
            else:
                raise ValueError(f"Provided n_rois={n_rois} does not match data shape {x.shape} for {path}")
    return x  # shape (T, R)


def compute_correlation(timeseries: np.ndarray):
    # timeseries: (T, R)
    corr = np.corrcoef(timeseries.T)
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)
    return corr


def compute_partial_correlation(timeseries: np.ndarray, reg: float = 1e-5):
    # Compute precision matrix from covariance with regularization
    cov = np.cov(timeseries.T)
    cov += np.eye(cov.shape[0]) * reg
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(prec))
    outer = np.outer(d, d)
    pcorr = -prec / outer
    np.fill_diagonal(pcorr, 0.0)
    pcorr = np.nan_to_num(pcorr)
    return pcorr


def sparsify_adj(adj: np.ndarray, top_k: int = None, top_percent: float = None, abs_thresh: float = None):
    A = adj.copy()
    R = A.shape[0]
    absA = np.abs(A)

    if abs_thresh is not None:
        mask = absA >= abs_thresh
        A[~mask] = 0.0
    elif top_k is not None:
        if top_k <= 0:
            return A
        np.fill_diagonal(absA, 0.0)
        flat = absA.flatten()
        kth = np.partition(flat, -top_k)[-top_k]
        mask = absA >= kth
        A[~mask] = 0.0
    elif top_percent is not None:
        if not (0 < top_percent <= 100):
            raise ValueError("top_percent must be in (0,100]")
        np.fill_diagonal(absA, 0.0)
        # number of undirected edges excluding diagonal
        num_possible = R * (R - 1) // 2
        k = int(np.ceil(num_possible * top_percent / 100.0))
        if k <= 0:
            return A
        flat = absA.flatten()
        kth = np.partition(flat, -k)[-k]
        mask = absA >= kth
        A[~mask] = 0.0

    # ensure symmetry & zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def make_node_features(timeseries: np.ndarray, mode: str = 'stats', pca_n: int = 8):
    T, R = timeseries.shape
    if mode == 'stats':
        means = np.mean(timeseries, axis=0)
        stds = np.std(timeseries, axis=0)
        feats = np.stack([means, stds], axis=1)  # (R,2)
    elif mode == 'timeseries':
        feats = timeseries.T  # (R, T)
    elif mode == 'pca':
        data = timeseries.T  # (R, T)
        n_comp = min(pca_n, min(data.shape))
        pca = PCA(n_components=n_comp)
        feats = pca.fit_transform(data)
    else:
        raise ValueError(f"Unknown node feature mode: {mode}")
    return feats


def adj_to_edge_index_and_attr(adj: np.ndarray, keep_self: bool = False):
    # Convert symmetric adj matrix -> edge_index (2, E) and edge_attr (E,)
    # Keep only upper triangle (i<j) and then duplicate to make undirected edges for PyG
    R = adj.shape[0]
    rows, cols = np.triu_indices(R, k=0 if keep_self else 1)
    weights = adj[rows, cols]
    # keep non-zero
    nz_mask = weights != 0
    rows = rows[nz_mask]
    cols = cols[nz_mask]
    weights = weights[nz_mask]

    if len(rows) == 0:
        # no edges, return empty tensors
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float32)
        return edge_index, edge_attr

    # form undirected edges by adding both directions
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index_np = np.vstack([src, dst])  # shape (2, 2E)
    edge_attr_np = np.concatenate([weights, weights])

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    return edge_index, edge_attr


def process_file(inpath: Path, outpath: Path, edge_method: str='correlation', sparsify_kwargs=None, node_mode: str='stats', pca_n: int = 8, meta: dict=None, keep_self: bool=False, n_rois: int=None):
    x = load_fmri(inpath, n_rois=n_rois)  # (T, R)
    T, R = x.shape

    if edge_method == 'correlation':
        adj = compute_correlation(x)
    elif edge_method == 'partial':
        adj = compute_partial_correlation(x)
    else:
        raise ValueError(f"Unknown edge_method: {edge_method}")

    sparsify_kwargs = sparsify_kwargs or {}
    adj = sparsify_adj(adj, **sparsify_kwargs)

    node_feats = make_node_features(x, mode=node_mode, pca_n=pca_n)  # (R, F) or (R, T)

    edge_index, edge_attr = adj_to_edge_index_and_attr(adj, keep_self=keep_self)

    out = {
        'adj': adj.astype(np.float32),
        'node_features': node_feats.astype(np.float32),
        'edge_index': edge_index.numpy(),
        'edge_attr': edge_attr.numpy(),
        'meta': {
            'orig_shape': (int(T), int(R)),
            'edge_method': edge_method,
            'sparsify': sparsify_kwargs,
            'node_mode': node_mode,
            'keep_self_loops_in_edge_index': bool(keep_self),
        }
    }
    if meta:
        out['meta'].update(meta)

    if os.path.exists(outpath):
        raise FileExistsError(f"Warning: Overwriting existing file {outpath}")
    np.save(str(outpath), out)
    return out


def iter_process(input_dir: Path, output_dir: Path, **kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(sorted(input_dir.glob('*.npy')))
    edge_method = kwargs.get('edge_method', 'correlation')
    unique_suffix = 'corr' if edge_method == 'correlation' else 'pcorr'
    if not files:
        raise FileNotFoundError(f'No .npy files found in {input_dir}')
    for f in tqdm(files, desc="Processing fMRI files"):
        name = f.stem
        outname = output_dir / f"{name}_{unique_suffix}_graph.npy"
        print(f"Processing {f.name} -> {outname.name}")
        process_file(f, outname, **kwargs)


class NPYGraphDataset(Dataset):
    """
    PyG Dataset that reads .npy graph files produced by this script and returns torch_geometric.data.Data objects.

    Args:
        root (str or Path): directory containing .npy files (or a parent directory); this class will look for any *.npy files.
        transform, pre_transform: same as PyG Dataset API (passed to super)
        use_adj_as_x (bool): 如果 True，则 Data.x 会被设置为 adjacency 矩阵（R, R）；否则使用保存的 node_features (R, F)
        label_map (dict): 可选， mapping from filename -> label (int) 用于设置 data.y
    """

    def __init__(self, root, transform=None, pre_transform=None, use_adj_as_x=False, label_map=None):
        super().__init__(root, transform, pre_transform)
        self.root = Path(root)
        # find files
        self.files = sorted(list(self.root.glob('*_graph.npy')) + [p for p in self.root.glob('*.npy') if not p.name.endswith('_graph.npy')])
        # filter out any non-graph npy? We assume the directory contains only our generated files or raw ones
        self.use_adj_as_x = use_adj_as_x
        self.label_map = label_map or {}

    def len(self):
        return len(self.files)

    def get(self, idx):
        fpath = self.files[idx]
        data_dict = np.load(str(fpath), allow_pickle=True).item()
        adj = data_dict.get('adj')
        node_feats = data_dict.get('node_features')
        edge_index_np = data_dict.get('edge_index')
        edge_attr_np = data_dict.get('edge_attr')

        # build tensors
        if self.use_adj_as_x:
            x = torch.tensor(adj, dtype=torch.float32)  # (R, R)
        else:
            x = torch.tensor(node_feats, dtype=torch.float32)
        if edge_index_np is None or edge_attr_np is None:
            # fallback: build from adj
            edge_index, edge_attr = adj_to_edge_index_and_attr(adj)
        else:
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # also attach full adjacency if needed
        data.adj = torch.tensor(adj, dtype=torch.float32)

        # try to set y if available in label_map
        basename = fpath.stem
        label = self.label_map.get(basename)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        else:
            data.y = None

        # attach metadata
        data.meta = data_dict.get('meta', {})
        return data


def parse_args():
    p = argparse.ArgumentParser(description='Convert time x roi .npy fMRI into graph .npy for PyG and provide a PyG Dataset class')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dir', type=str, help='Directory with .npy files (each is time x roi)')
    group.add_argument('--input_file', type=str, help='Single .npy file to process')
    group.add_argument('--input_dirs', type=str, nargs='+', help='Multiple input directories to process sequentially')
    p.add_argument('--output_dir', type=str, default='./graphs_npy', help='Where to save graph npy files')
    p.add_argument('--n_rois', type=int, help='Number of ROIs, if not provided will be inferred from data, and the smaller dimension of the input array is assumed to be ROIs')
    p.add_argument('--edge_method', choices=['correlation', 'partial'], default='correlation')
    p.add_argument('--abs_thresh', type=float, help='Absolute threshold for edges (applied to abs value)')
    p.add_argument('--top_k', type=int, help='Keep top_k strongest edges (by absolute weight)')
    p.add_argument('--top_percent', type=float, help='Keep top percent of edges (0-100)')
    p.add_argument('--node_mode', choices=['stats', 'timeseries', 'pca'], default='stats')
    p.add_argument('--pca_n', type=int, default=8, help='Number of PCA components when node_mode==pca')
    p.add_argument('--label', type=str, help='Optional label to store in meta')
    p.add_argument('--keep_self', action='store_true', help='Whether to keep self-loops when creating edge_index (default False)')
    return p.parse_args()


def main():
    args = parse_args()
    sparsify = {}
    if args.abs_thresh is not None:
        sparsify['abs_thresh'] = float(args.abs_thresh)
    if args.top_k is not None:
        sparsify['top_k'] = int(args.top_k)
    if args.top_percent is not None:
        sparsify['top_percent'] = float(args.top_percent)

    meta = {}
    if args.label is not None:
        meta['label'] = args.label

    if args.input_file:
        edge_method = args.edge_method
        unique_suffix = 'corr' if edge_method == 'correlation' else 'pcorr'

        inp = Path(args.input_file)
        out = Path(args.output_dir) / f"{inp.stem}_{unique_suffix}_graph.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Processing single file {inp} -> {out}")
        process_file(inp, out, edge_method=args.edge_method, sparsify_kwargs=sparsify, node_mode=args.node_mode, pca_n=args.pca_n, meta=meta, keep_self=args.keep_self, n_rois=args.n_rois)
    elif args.input_dir:
        iter_process(Path(args.input_dir), Path(args.output_dir), edge_method=args.edge_method, sparsify_kwargs=sparsify, node_mode=args.node_mode, pca_n=args.pca_n, meta=meta, keep_self=args.keep_self, n_rois=args.n_rois)
    elif args.input_dirs:
        for indir in args.input_dirs:
            print(f"Processing directory {indir}")
            iter_process(Path(indir), Path(args.output_dir), edge_method=args.edge_method, sparsify_kwargs=sparsify, node_mode=args.node_mode, pca_n=args.pca_n, meta=meta, keep_self=args.keep_self, n_rois=args.n_rois)


if __name__ == '__main__':
    main()

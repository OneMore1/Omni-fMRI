# -*- coding: utf-8 -*-
# clip_tsne_color.py
"""
Reads images from the training directory and its subdirectories.
Extracts features using OpenAI's official CLIP model (either backbone or projection),
performs PCA followed by t-SNE for dimensionality reduction, and visualizes the results.

Images with the parent directory name 'shared_1120' are plotted in a distinct color
to distinguish them from other directories.

Dependencies:
    pip install "git+https://github.com/openai/CLIP.git" torch torchvision pillow scikit-learn matplotlib tqdm packaging
"""

# -- Prevent OpenBLAS thread oversubscription (Recommended for high-core-count machines) --
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import glob
import csv
import argparse
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import clip
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn, inspect
from packaging.version import parse as V

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========== Dataset ==========
class ImagePathDataset(Dataset):
    def __init__(self, img_dir: str, preprocess, exts=(".jpg", ".jpeg", ".png", ".webp")):
        paths = []
        for e in exts:
            # Search recursively
            paths += glob.glob(os.path.join(img_dir, f"*{e}"))
            paths += glob.glob(os.path.join(img_dir, "*", f"*{e}"))
            paths += glob.glob(os.path.join(img_dir, "*", "*", f"*{e}"))
        self.paths = sorted(list(set(paths)))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {img_dir} (supported extensions: {exts})")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        # Extract the parent directory name as a "group" label
        # (Used specifically to distinguish 'shared_1120' from others)
        parent = os.path.basename(os.path.dirname(p))
        label = 1 if parent == "shared_1120" else 0
        return self.preprocess(img), p, label


# ========== Utils ==========
def l2norm(x: np.ndarray, eps=1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


# ========== Feature Extraction ==========
@torch.no_grad()
def extract_features(
    train_dir: str,
    model_name: str = "ViT-B/32",
    feature_type: str = "backbone",   # 'backbone' or 'proj'
    batch_size: int = 256,
    num_workers: int = 8,
    use_fp16: bool = False,
    max_images: int = 0,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Unify data type strategy to avoid inconsistencies between half and single precision
    if device == "cuda":
        if use_fp16:
            model.half()
        else:
            model.float()
    else:
        model.float()

    ds = ImagePathDataset(train_dir, preprocess)
    if max_images and max_images < len(ds):
        ds.paths = ds.paths[:max_images]
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, shuffle=False)

    feats, names, labels = [], [], []

    pbar = tqdm(dl, total=len(dl), desc=f"Extract ({model_name}|{feature_type})")
    target_dtype = next(model.parameters()).dtype
    for imgs, paths, labs in pbar:
        imgs = imgs.to(device, non_blocking=True).to(dtype=target_dtype)
        if feature_type == "proj":
            emb = model.encode_image(imgs)  # Semantic features after projection
        else:
            emb = model.visual(imgs)        # Backbone pooled features (before projection)
        
        emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
        feats.append(emb); names += list(paths); labels.append(labs.numpy())

    feats = np.concatenate(feats, axis=0)        # (N, D)
    labels = np.concatenate(labels, axis=0)      # (N,)
    return feats, names, labels


# ========== t-SNE ==========
def run_tsne(
    feats: np.ndarray,
    names: List[str],
    labels: np.ndarray,           # 1: shared_1120; 0: others
    out_prefix: str = "tsne",
    pca_dim: int = 50,
    tsne_perplexity: float = 20.0,
    tsne_iter: int = 1500,
    metric: str = "cosine",
    random_state: int = 0,
):
    N, D = feats.shape
    X = feats.astype("float32")

    # PCA
    if pca_dim and pca_dim < D:
        print(f"[PCA] {D} -> {pca_dim}")
        Xp = PCA(n_components=pca_dim, random_state=random_state, svd_solver="auto").fit_transform(X)
    else:
        Xp = X

    print(f"[t-SNE] N={N}, dim={Xp.shape[1]}, perplexity={tsne_perplexity}, iters={tsne_iter}, metric={metric}")
    
    # Handle scikit-learn version compatibility
    ver = V(sklearn.__version__)
    base_kwargs = dict(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        metric=metric,
        random_state=random_state,
        verbose=1,
    )
    if ver >= V("1.6"):
        base_kwargs["max_iter"] = tsne_iter
    else:
        base_kwargs["n_iter"] = tsne_iter
        
    if ver >= V("1.2"):
        base_kwargs["learning_rate"] = "auto"
        base_kwargs["square_distances"] = True
    else:
        base_kwargs["learning_rate"] = 200
        
    sig = inspect.signature(TSNE.__init__)
    if "n_jobs" in sig.parameters:
        base_kwargs["n_jobs"] = 1
    tsne_kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}

    Y = TSNE(**tsne_kwargs).fit_transform(Xp)  # (N, 2)

    # -- Save Data Tables -- #
    np.save(f"{out_prefix}_feats.npy", X)
    with open(f"{out_prefix}_paths.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path","label"])
        for p, lb in zip(names, labels): w.writerow([p, int(lb)])
    with open(f"{out_prefix}_2d.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x","y","label","path"])
        for (x,y), lb, p in zip(Y, labels, names):
            w.writerow([float(x), float(y), int(lb), p])

    # -- Plot Scatter (Two Colors) -- #
    idx_shared = labels == 1
    idx_other  = ~idx_shared
    plt.figure(figsize=(8,8), dpi=160)
    plt.scatter(Y[idx_other,0],  Y[idx_other,1],  s=6, alpha=0.6, label="others")
    plt.scatter(Y[idx_shared,0], Y[idx_shared,1], s=12, alpha=0.8, label="shared_1120")
    plt.legend(loc="best")
    plt.title("t-SNE (CLIP features): shared_1120 vs others")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_2d.png"); plt.close()
    print(f"✅ Saved: {out_prefix}_2d.png / {out_prefix}_2d.csv / {out_prefix}_paths.csv / {out_prefix}_feats.npy")


# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, help="Root directory of training images (includes subdirectories)")
    ap.add_argument("--model", default="ViT-B/32", help="OpenAI CLIP model name: ViT-B/32, ViT-L/14, RN50, etc.")
    ap.add_argument("--feature-type", default="backbone", choices=["backbone","proj"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--max-images", type=int, default=0, help="Extract features for only the first N images for visualization; 0 = no limit")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=20.0)
    ap.add_argument("--tsne-iter", type=int, default=1500)
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])
    ap.add_argument("--out-prefix", default="tsne_color")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    feats, names, labels = extract_features(
        train_dir=args.train_dir,
        model_name=args.model,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fp16=args.use_fp16,
        max_images=args.max_images,
        device=device
    )
    feats = l2norm(feats)

    run_tsne(
        feats=feats,
        names=names,
        labels=labels,
        out_prefix=args.out_prefix,
        pca_dim=args.pca_dim,
        tsne_perplexity=args.perplexity,
        tsne_iter=args.tsne_iter,
        metric=args.metric,
        random_state=0
    )

if __name__ == "__main__":
    main()

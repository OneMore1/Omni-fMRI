#!/usr/bin/env python3
"""Smoke-test the Omni-fMRI quick start with synthetic fMRI data.

Run this script from the root of the Omni-fMRI GitHub repository. It creates a
small synthetic NIfTI file, preprocesses it with the repository preprocessing
script, extracts features with the released checkpoint, and verifies the output
token arrays.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrain_checkpoint/checkpoint.pth"),
        help="Path to the Omni-fMRI checkpoint.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/quickstart_smoke"),
        help="Output directory for synthetic data and extracted features.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device passed to extract_feat.py, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke repository scripts.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_synthetic_nifti(raw_path: Path) -> None:
    import nibabel as nib
    import numpy as np

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    data = np.zeros((96, 96, 96, 40), dtype=np.float32)
    data[32:64, 30:66, 28:70, :] = rng.normal(
        0.0, 0.1, size=(32, 36, 42, 40)
    ).astype(np.float32)
    data[46:52, 44:54, 40:56, :] += np.linspace(
        -0.5, 0.5, 40, dtype=np.float32
    )
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), raw_path)
    print(f"wrote {raw_path} shape={data.shape} dtype={data.dtype}", flush=True)


def validate_tokens(feature_dir: Path) -> None:
    import numpy as np

    outputs = sorted(feature_dir.glob("*_tokens.npz"))
    if not outputs:
        raise FileNotFoundError(f"No *_tokens.npz files found in {feature_dir}")

    tokens = np.load(outputs[0])
    required = ["cls_token", "patch_tokens", "patch_coords"]
    for key in required:
        if key not in tokens:
            raise KeyError(f"Missing key {key!r} in {outputs[0]}")

    cls_token = tokens["cls_token"]
    patch_tokens = tokens["patch_tokens"]
    patch_coords = tokens["patch_coords"]

    if cls_token.ndim != 1:
        raise ValueError(f"cls_token should be 1D, got {cls_token.shape}")
    if patch_tokens.ndim != 2:
        raise ValueError(f"patch_tokens should be 2D, got {patch_tokens.shape}")
    if patch_coords.ndim != 2:
        raise ValueError(f"patch_coords should be 2D, got {patch_coords.shape}")

    print(f"validated {outputs[0]}")
    print(f"cls_token {cls_token.shape} {cls_token.dtype}")
    print(f"patch_tokens {patch_tokens.shape} {patch_tokens.dtype}")
    print(f"patch_coords {patch_coords.shape} {patch_coords.dtype}")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    if not (repo_root / "extract_feat.py").is_file():
        raise FileNotFoundError("Run this script from the Omni-fMRI repository root.")
    if not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    work_dir = args.work_dir.expanduser().resolve()
    raw_dir = work_dir / "raw"
    processed_dir = work_dir / "processed_npz"
    feature_dir = work_dir / "features"

    raw_path = raw_dir / "sample_fmri.nii.gz"
    write_synthetic_nifti(raw_path)

    run_command(
        [
            args.python,
            "data_preparation/preprocessing.py",
            "-i",
            str(raw_dir),
            "-o",
            str(processed_dir),
            "--pattern",
            ".nii.gz",
            "--target_shape",
            "96",
            "96",
            "96",
            "-s",
            "40",
        ],
        cwd=repo_root,
    )

    run_command(
        [
            args.python,
            "extract_feat.py",
            str(processed_dir),
            "--checkpoint",
            str(args.checkpoint.expanduser().resolve()),
            "--output-dir",
            str(feature_dir),
            "--device",
            args.device,
            "--overwrite",
        ],
        cwd=repo_root,
    )

    validate_tokens(feature_dir)
    print(f"QUICKSTART_SMOKE_PASS {work_dir}", flush=True)


if __name__ == "__main__":
    main()

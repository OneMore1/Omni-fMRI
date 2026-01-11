#!/usr/bin/env python3
'''
Author: ViolinSolo
Date: 2025-10-31 15:37:04
LastEditTime: 2025-11-11 15:24:25
LastEditors: ViolinSolo
Description: 

Scan a dataset root for .npy/.npz files containing fMRI time series arrays shaped (T, R).
For each file:
 - take the first `time_length` time points (T_slice = min(time_length, T))
 - compute Pearson correlation between ROIs -> adjacency matrix (R x R)
 - save adjacency matrix to destination directory, preserving relative path and filename
 - record processing info into a CSV file

Features:
 - argparse CLI
 - tqdm progress bar
 - optional multiprocessing for parallel processing
 - robust loading for .npy and .npz
 - handles constant time series (replace NaN correlations with 0)
 - writes a CSV summary at the end

=================================
python make_adjacency.py --src /path/to/your/dataset --dest /path/to/save/adj --time_length 120 --workers 8 --save_format npz --csv summaries.csv --overwrite
=================================

FilePath: /ProjectBrainBaseline/scripts/make_adj_martrix.py
'''


import os
import sys
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import numpy as np
from tqdm import tqdm
import csv
import datetime

# ---------------------------
# Helper functions
# ---------------------------

def find_np_files(root):
    """
    Recursively find .npy and .npz files under root.
    Returns a list of absolute file paths (strings).
    """
    root = Path(root)
    files = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.npy', '.npz'):
            files.append(str(p))
    return files

def load_array(file_path, transpose=False):
    """
    Load array from .npy or .npz. For .npz, try to find the first array-like value.
    Returns numpy.ndarray.
    Raises ValueError if cannot load or shape is unexpected.
    """
    file_path = str(file_path)
    suffix = file_path.lower().split('.')[-1]
    if suffix == 'npy':
        arr = np.load(file_path, allow_pickle=False)
        return arr.T if transpose else arr
    elif suffix == 'npz':
        try:
            npz = np.load(file_path, allow_pickle=False)
            # npz is an NpzFile which acts like a dict of arrays.
            # Prefer arrays by key order; if only one array, return it.
            keys = list(npz.keys())
            if len(keys) == 0:
                raise ValueError('Empty npz archive')
            # Try to find the first array with ndim >= 2
            for k in keys:
                a = npz[k]
                if isinstance(a, np.ndarray):
                    return a
            # fallback: return first
            return npz[keys[0]].T if transpose else npz[keys[0]]
        finally:
            # npz is closed automatically on garbage collect, but ensure deletion
            try:
                npz.close()
            except Exception:
                pass
    else:
        raise ValueError(f'Unsupported file suffix: {suffix}')

def compute_adjacency(time_series, time_length):
    """
    Given time_series shaped (T, R), slice to first time_length rows and compute
    Pearson correlation between columns (ROIs).
    Returns adjacency matrix of shape (R, R).
    """
    if time_series.ndim != 2:
        raise ValueError(f"Expected 2D array (T, R), got shape {time_series.shape}")
    T, R = time_series.shape
    t_use = min(time_length, T)
    if t_use < 2:
        raise ValueError("Need at least 2 time points to compute correlation")
    data = time_series[:t_use, :]  # shape (t_use, R)
    # Compute Pearson correlation between columns -> use np.corrcoef with rowvar=False
    # This returns R x R matrix.
    # Note: if a column is constant, corrcoef will produce NaN; we replace NaN with 0.
    with np.errstate(invalid='ignore'):
        adj = np.corrcoef(data, rowvar=False)
    # Replace NaN (from zero std columns) with 0
    if np.isnan(adj).any():
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    return adj

def ensure_dest_path(src_path, src_root, dest_root):
    """
    Given source file path and source root, create a corresponding destination directory
    under dest_root that preserves relative path. Return the destination base directory (Path).
    """
    src_path = Path(src_path)
    src_root = Path(src_root)
    dest_root = Path(dest_root)
    rel = src_path.relative_to(src_root)
    dest_dir = dest_root.joinpath(rel.parent)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir

def process_one(file_path, src_root, dest_root, time_length, overwrite, save_format, transpose):
    """
    Process a single file: load, compute adjacency, save, and return a dict record.
    This function is designed to be picklable for multiprocessing.
    """
    record = {
        'src_path': str(file_path),
        'rel_path': None,
        'orig_shape': None,
        'used_time_length': None,
        'adj_shape': None,
        'saved_path': None,
        'status': 'ok',
        'message': '',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }
    try:
        array = load_array(file_path, transpose=transpose)
        if not isinstance(array, np.ndarray):
            raise ValueError("Loaded object is not a numpy array")
        record['orig_shape'] = str(array.shape if not transpose else array.T.shape)  # resume original shape
        # ensure 2D
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array (T,R), got ndim={array.ndim}")
        # compute adjacency
        adj = compute_adjacency(array, time_length)
        record['used_time_length'] = min(time_length, array.shape[0])
        record['adj_shape'] = str(adj.shape)
        # determine save path (preserve relative path, same base name)
        rel = Path(file_path).relative_to(Path(src_root))
        record['rel_path'] = str(rel)
        dest_dir = ensure_dest_path(file_path, src_root, dest_root)
        base_name = Path(file_path).stem  # without suffix
        # choose save filename and format
        if save_format == 'npy':
            out_name = base_name + '.npy'
            out_path = dest_dir.joinpath(out_name)
            if out_path.exists() and not overwrite:
                record['status'] = 'skipped'
                record['message'] = 'exists'
                record['saved_path'] = str(out_path)
                return record
            np.save(str(out_path), adj)
        elif save_format == 'npz':
            out_name = base_name + '.npz'
            out_path = dest_dir.joinpath(out_name)
            if out_path.exists() and not overwrite:
                record['status'] = 'skipped'
                record['message'] = 'exists'
                record['saved_path'] = str(out_path)
                return record
            # save compressed
            np.savez_compressed(str(out_path), adjacency=adj)
        else:
            raise ValueError(f"Unknown save_format: {save_format}")
        record['saved_path'] = str(out_path)
    except Exception as e:
        record['status'] = 'error'
        record['message'] = repr(e)
        # include traceback for debug (short)
        tb = traceback.format_exc(limit=1)
        record['message'] += ' | tb:' + tb.strip().splitlines()[-1] if tb else ''
    return record

# ---------------------------
# CLI / Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert fMRI time-series (.npy/.npz with shape (T,R)) into adjacency matrices (Pearson).")
    p.add_argument('--src', '-s', required=True, help='Root directory of dataset (will be searched recursively).')
    p.add_argument('--dest', '-d', required=True, help='Destination root directory to save adjacency matrices (preserve relative paths).')
    p.add_argument('--time_length', '-t', type=int, required=True, help='Number of initial time points to use from each file (slice length). ')
    p.add_argument('--workers', '-w', type=int, default=1, help='Number of parallel worker processes (default 1).')
    p.add_argument('--csv', default='adjacency_records.csv', help='Output CSV filename to record processing info (default adjacency_records.csv in dest).')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files if present.')
    p.add_argument('--save_format', choices=['npy', 'npz'], default='npy', help='Save adjacency as .npy or compressed .npz (default npy).')
    p.add_argument('--min_timepoints', type=int, default=2, help='Minimum timepoints required to compute correlation (default 2).')
    p.add_argument('--transpose', action='store_true', help='If set, transpose loaded arrays (assume shape (R, T) instead of (T, R)).')
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest.joinpath(args.csv)

    # find files
    files = find_np_files(src)
    if len(files) == 0:
        print(f"No .npy or .npz files found under {src}")
        return

    # Prepare worker func
    process_func = partial(
        process_one,
        src_root=str(src),
        dest_root=str(dest),
        time_length=args.time_length,
        overwrite=args.overwrite,
        save_format=args.save_format,
        transpose=args.transpose
    )

    records = []
    # Use multiprocessing Pool if workers > 1
    if args.workers > 1:
        # Use imap_unordered to stream results and update tqdm
        with Pool(processes=args.workers) as pool:
            for rec in tqdm(pool.imap_unordered(process_func, files), total=len(files), desc='Processing', unit='file'):
                records.append(rec)
    else:
        # single-process, but still show tqdm
        for f in tqdm(files, desc='Processing', unit='file'):
            rec = process_func(f)
            records.append(rec)

    # Write CSV summary
    csv_fields = ['src_path', 'rel_path', 'orig_shape', 'used_time_length', 'adj_shape', 'saved_path', 'status', 'message', 'timestamp']
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            for r in records:
                # ensure all keys exist
                row = {k: r.get(k, '') for k in csv_fields}
                writer.writerow(row)
        print(f"Wrote CSV summary to: {csv_path}")
    except Exception as e:
        print(f"Failed writing CSV: {e}")

    # Print simple summary counts
    total = len(records)
    ok = sum(1 for r in records if r['status'] == 'ok')
    skipped = sum(1 for r in records if r['status'] == 'skipped')
    errors = sum(1 for r in records if r['status'] == 'error')
    print(f"Done. Total {total}, ok={ok}, skipped={skipped}, errors={errors}")

if __name__ == '__main__':
    main()

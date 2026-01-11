#!/usr/bin/env python3
'''
Author: ViolinSolo
Date: 2025-11-07 15:03:35
LastEditTime: 2025-11-27 17:53:27
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/scripts/trains/collect_results.py
'''
"""
collect_results.py

Usage:
    python collect_results.py /path/to/experiments_root -o combined_results.csv

What it does:
 - Walks the given root directory and finds subdirectories that contain a config.yaml and/or test_results.csv
 - Extracts batch_size, lr, wd, downstream_dataset from config.yaml (searches dict recursively)
 - Reads test_results.csv (expects columns 'accruacy' OR 'accuracy', and 'f1_score')
 - Produces a combined CSV with one row per experiment and columns:
     subdir, batch_size, lr, wd, downstream_dataset, accuracy, f1_score, config_path, results_path
 - Sorts rows by downstream_dataset, lr, wd, batch_size (ascending)
"""

import argparse
import os
import sys
import csv
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

# --- helpers -----------------------------------------------------------------

def find_value_recursive(d: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    """
    Recursively search a nested dict/list for any of the target keys.
    Return the first found value (depth-first) or None.
    keys: tuple of candidate key names to look for (case-insensitive).
    """
    if d is None:
        return None
    if isinstance(d, dict):
        for k, v in d.items():
            if any(k.lower() == cand.lower() for cand in keys):
                return v
            res = find_value_recursive(v, keys)
            if res is not None:
                return res
    elif isinstance(d, list):
        for item in d:
            res = find_value_recursive(item, keys)
            if res is not None:
                return res
    return None

def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def safe_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            f = float(x)
            return int(f)
        except Exception:
            return None

# --- main logic --------------------------------------------------------------

def process_experiment_dir(exp_dir: str) -> Optional[Dict]:
    """
    Process one experiment directory. Returns a dict or None if neither config nor results present.
    """
    config_path = None
    results_path = None

    # look for config.yaml (case-insensitive)
    for candidate in ["config.yaml", "config.yml", "Config.yaml", "CONFIG.YAML"]:
        p = os.path.join(exp_dir, candidate)
        if os.path.isfile(p):
            config_path = p
            break

    # look for test_results.csv (case-insensitive)
    for candidate in ["test_results.csv", "test-results.csv", "results.csv", "test_results.TSV"]:
        p = os.path.join(exp_dir, candidate)
        if os.path.isfile(p):
            results_path = p
            break

    if (config_path is None) and (results_path is None):
        return None

    batch_size = None
    lr = None
    wd = None
    downstream_dataset = None
    model = None
    accuracy = None
    f1_score = None
    mse = None
    r2_score = None

    # read config
    if config_path:
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            warnings.warn(f"Failed to parse YAML in {config_path}: {e}")
            cfg = None

        if cfg:
            batch_size = find_value_recursive(cfg, ("batch_size", "batchsize", "batch-size"))
            lr = find_value_recursive(cfg, ("lr", "learning_rate", "learning-rate", "learningrate"))
            wd = find_value_recursive(cfg, ("wd", "weight_decay", "weight-decay"))
            seed = find_value_recursive(cfg, ("seed",))
            downstream_dataset = find_value_recursive(cfg, ("downstream_dataset", "downstream_dataset_name", "dataset", "downstream")) if "unique_collect_str" not in cfg else cfg["unique_collect_str"]
            model = cfg.get("model", None).get("name", None) if isinstance(cfg.get("model", None), dict) else None
            downstream_task_type = cfg.get("task", None).get("name", None) if isinstance(cfg.get("task", None), dict) else None
            cls_label_col = find_value_recursive(cfg, ("label_col", "label_column", "target_column"))
            upstream_model = find_value_recursive(cfg, ("upstream_model", "pretrained_model", "upstream"))
            
            # try to coerce numeric fields
            batch_size = safe_int(batch_size)
            seed = safe_int(seed)
            lr = safe_float(lr)
            wd = safe_float(wd)

            
    # read results
    if results_path:
        try:
            df = pd.read_csv(results_path)
            # check for spelled columns: user said 'accruacy' and 'f1_score'
            if "accruacy" in df.columns:
                accuracy = df["accruacy"].iloc[-1]  # take last row by default
            elif "accuracy" in df.columns:
                accuracy = df["accuracy"].iloc[-1]
            elif "acc" in df.columns:
                accuracy = df["acc"].iloc[-1]
            else:
                warnings.warn(f"No accuracy-like column found in {results_path}. Available columns: {list(df.columns)}")

            # f1_score column
            if "f1_score" in df.columns:
                f1_score = df["f1_score"].iloc[-1]
            elif "f1" in df.columns:
                f1_score = df["f1"].iloc[-1]
            else:
                warnings.warn(f"No f1-like column found in {results_path}. Available columns: {list(df.columns)}")

            # mse column
            if "mse" in df.columns:
                mse = df["mse"].iloc[-1]
            else:
                warnings.warn(f"No mse column found in {results_path}. Available columns: {list(df.columns)}")
            # r2_score column
            if "r2_score" in df.columns:
                r2_score = df["r2_score"].iloc[-1]
            elif "r2" in df.columns:
                r2_score = df["r2"].iloc[-1]
            else:
                warnings.warn(f"No r2-like column found in {results_path}. Available columns: {list(df.columns)}")

            # coerce numeric if possible
            accuracy = safe_float(accuracy)
            f1_score = safe_float(f1_score)
            mse = safe_float(mse)
            r2_score = safe_float(r2_score)

        except Exception as e:
            warnings.warn(f"Failed to read results CSV {results_path}: {e}")

    return {
        "label_col": cls_label_col,
        "downstream_task_type": downstream_task_type,
        "downstream_dataset": downstream_dataset,
        "upstream_model": upstream_model,
        "model": model,
        "batch_size": batch_size,
        "lr": lr,
        "wd": wd,
        "seed": seed,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "mse": mse,
        "r2_score": r2_score,
        "subdir": os.path.relpath(exp_dir),
        "config_path": config_path,
        "results_path": results_path,
    }

def collect_all(root: str, recursive: bool = True):
    """
    Walk root directory and collect experiments.
    If recursive True, any directory at any depth is considered an 'experiment dir'.
    """
    rows = []
    # We'll treat any directory that contains config.yaml or test_results.csv as an experiment.
    for dirpath, dirnames, filenames in os.walk(root):
        # skip root itself if you want immediate subdirs only, but user asked "iterate under an directory" - we'll include nested
        res = process_experiment_dir(dirpath)
        if res:
            rows.append(res)
    return rows

def save_combined(rows, out_path):
    if not rows:
        print("No experiments found. Nothing to save.")
        return
    df = pd.DataFrame(rows)

    # Normalize downstream_dataset as string for sorting
    df["downstream_dataset"] = df["downstream_dataset"].astype(str)

    # Fill NaNs with something predictable for sorting (put None/NaN after actual values)
    # We'll sort with NaNs last. Pandas sorts NaN as last when na_position='last' only for sort_values param.
    sort_cols = ["upstream_model", "label_col", "downstream_task_type", "downstream_dataset", "model", "seed", "lr", "wd", "batch_size"]
    # ensure numeric columns exist
    for col in ["lr", "wd", "batch_size"]:
        if col not in df.columns:
            df[col] = pd.NA

    df_sorted = df.sort_values(by=sort_cols, ascending=[True, True, True, True, True, True, True, True, True], na_position="last").reset_index(drop=True)

    df_sorted.to_csv(out_path, index=False)
    print(f"Saved combined results to {out_path} ({len(df_sorted)} rows).")

    # calculate the average accuracy and f1_score per downstream_dataset and model, and its corresponding standard deviation
    summary = df_sorted.groupby(["upstream_model", "label_col", "downstream_dataset", "model"]).agg(
        accuracy_mean=pd.NamedAgg(column="accuracy", aggfunc="mean"),
        accuracy_std=pd.NamedAgg(column="accuracy", aggfunc="std"),
        f1_score_mean=pd.NamedAgg(column="f1_score", aggfunc="mean"),
        f1_score_std=pd.NamedAgg(column="f1_score", aggfunc="std"),
        mse_mean=pd.NamedAgg(column="mse", aggfunc="mean"),
        mse_std=pd.NamedAgg(column="mse", aggfunc="std"),
        r2_score_mean=pd.NamedAgg(column="r2_score", aggfunc="mean"),
        r2_score_std=pd.NamedAgg(column="r2_score", aggfunc="std"),
    ).reset_index() 
    summary_path = os.path.splitext(out_path)[0] + "_stats.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to {summary_path}.")

# --- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Collect experiment results into a combined CSV.")
    p.add_argument("root", help="Root directory containing experiment subdirectories.")
    p.add_argument("-o", "--output", default="combined_results.csv", help="Path to output CSV file.")
    p.add_argument("--non-recursive", action="store_true", help="Do not search nested directories; only inspect immediate subdirectories.")
    return p.parse_args()

def main():
    args = parse_args()
    root = args.root
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    if args.non_recursive:
        # only immediate subdirs
        rows = []
        for name in os.listdir(root):
            sub = os.path.join(root, name)
            if os.path.isdir(sub):
                r = process_experiment_dir(sub)
                if r:
                    rows.append(r)
    else:
        rows = collect_all(root, recursive=True)

    save_combined(rows, args.output)

if __name__ == "__main__":
    main()

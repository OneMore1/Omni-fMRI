#!/usr/bin/env python3
"""
tr_resample_npy.py

遍历源目录下所有 .npy 文件，根据给定 TR（原始 TR）对每个文件做时间轴重插值（CubicSpline），
并在目标目录按相同层级结构保存插值后的数据。

特点:
- 支持多线程（ThreadPoolExecutor）
- 控制台进度条（tqdm）
- 进度记录到 JSON 文件（progress.json），便于断点/恢复
- 支持跳过已存在输出（--resume）和覆盖输出（--overwrite）

示例:
python tr_resample_npy.py --src /path/to/src --dst /path/to/dst --tr-original 2.0 --new-tr 0.72 --workers 8

python scripts/datasets/tr_resample_npy.py --src data/cvpr_dataset/fmri_ROI/ppmi --dst data/cvpr_dataset/ppmi_tr072 --tr-original 2.5 --new-tr 0.72 --workers 8
python scripts/datasets/tr_resample_npy.py --src data/cvpr_dataset/fmri_ROI/adni --dst data/cvpr_dataset/adni_tr072 --tr-original 3 --new-tr 0.72 --workers 8

注意:
- 假设每个 npy 文件的数组形状为 (ROI, T)，即 ROI x 时间。
- 依赖: numpy, scipy, tqdm
"""

import argparse
import logging
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def find_npy_files(root: Path):
    return [p for p in root.rglob('*.npy') if p.is_file()]


def ensure_parent(dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)


def process_file(src_path: Path, dst_path: Path, tr_original: float, new_tr: float):
    """读入 src_path 的 npy，按用户给定 TR 做插值，保存到 dst_path。
    返回 (src_path, dst_path, success, msg)
    """
    try:
        data = np.load(src_path)
        if data.ndim != 2:
            return (str(src_path), str(dst_path), False, f"数据维度不是 2D，而是 {data.ndim}D")
        ROI, T = data.shape
        # 构建时间点。使用 arange 的形式与用户提供的代码一致。
        original_times = np.arange(0, T * tr_original, tr_original)
        # 某些情况下 due to floating point 末尾长度可能与 T mismatch -> force length = T
        if original_times.shape[0] != T:
            # fallback: 使用索引乘 TR
            original_times = np.arange(T) * tr_original
        new_times = np.arange(0, T * tr_original, new_tr)

        interpolated = np.zeros((ROI, len(new_times)), dtype=np.float32)
        for i in range(ROI):
            cs = CubicSpline(original_times, data[i, :])
            interpolated[i, :] = cs(new_times)

        ensure_parent(dst_path)
        # 保存为 float32，节约空间
        np.save(dst_path, interpolated)
        return (str(src_path), str(dst_path), True, f"OK, shape {interpolated.shape}")
    except Exception as e:
        return (str(src_path), str(dst_path), False, repr(e))


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description='遍历 npy 并按 TR 重插值（CubicSpline）')
    parser.add_argument('--src', required=True, help='源根目录，递归查找 .npy')
    parser.add_argument('--dst', required=True, help='目标根目录，按相同层级保存插值结果')
    parser.add_argument('--tr-original', type=float, required=True, help='原始 TR（单位：s），例如 2.0')
    parser.add_argument('--new-tr', type=float, default=0.72, help='新的 TR（单位：s），默认 0.72')
    parser.add_argument('--workers', type=int, default=4, help='并发线程数，默认 4')
    parser.add_argument('--resume', action='store_true', help='如果目标文件已存在则跳过（用于断点续跑）')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的目标文件')
    parser.add_argument('--progress-file', default='progress.json', help='记录进度的 JSON 文件路径')
    parser.add_argument('--log', default=None, help='可选日志文件路径')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log, format='[%(asctime)s] %(levelname)s: %(message)s')
    console = logging.getLogger()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"源目录不存在: {src_root}")

    files = find_npy_files(src_root)
    total = len(files)
    logging.info(f'Found {total} .npy files under {src_root}')

    # Prepare progress structure
    progress_lock = threading.Lock()
    progress_path = Path(args.progress_file)
    if progress_path.exists() and args.resume:
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
        except Exception:
            progress = {'total': total, 'done': 0, 'files': {}}
    else:
        progress = {'total': total, 'done': 0, 'files': {}}
        atomic_write_json(progress_path, progress)

    # prepare tasks
    tasks = []
    for f in files:
        rel = f.relative_to(src_root)
        dst = dst_root / rel
        tasks.append((f, dst))

    # Optionally filter tasks when resume and files already exist
    if args.resume:
        filtered = []
        for src, dst in tasks:
            if dst.exists():
                # skip
                progress['files'][str(src)] = {'status': 'skipped_exists'}
            else:
                filtered.append((src, dst))
        tasks = filtered
        # refresh total to remaining for the run (but keep original total in progress)
    
    # Run with ThreadPoolExecutor and tqdm
    pbar = tqdm(total=len(tasks), desc='Processing', unit='file')

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_task = {ex.submit(process_file, src, dst, args.tr_original, args.new_tr): (src, dst) for src, dst in tasks}
        for future in as_completed(future_to_task):
            src, dst = future_to_task[future]
            try:
                src_p, dst_p, success, msg = future.result()
            except Exception as e:
                src_p, dst_p, success, msg = (str(src), str(dst), False, repr(e))
            with progress_lock:
                progress['done'] = progress.get('done', 0) + 1
                progress['files'][str(src)] = {'dst': str(dst), 'success': bool(success), 'msg': msg}
                try:
                    atomic_write_json(progress_path, progress)
                except Exception as e:
                    logging.warning(f'写入进度文件失败: {e}')
            pbar.update(1)
            if not success:
                logging.error(f'Failed: {src} -> {dst} : {msg}')
            else:
                logging.info(f'OK: {src} -> {dst} : {msg}')
    pbar.close()

    logging.info('All done')
    print(f"Finished. Progress recorded in {progress_path}")


if __name__ == '__main__':
    main()

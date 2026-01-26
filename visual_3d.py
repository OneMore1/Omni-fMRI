# ---------------------------------------------------------------
# References:
# APT: https://github.com/rccchoudhury/apt
# ---------------------------------------------------------------

import sys
sys.path.append("..")
import torch
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image

from src.models.entropy_utils_3d import (
    compute_patch_entropy_batched_3d, 
    compute_patch_laplacian_batched_3d,
    compute_patch_mse_batched_3d,
    select_patches_by_threshold_3d,
    compute_patch_local_std_batched_3d,
)

# Default parameters
IMAGE_SIZE = (96, 96, 96) 
BASE_PATCH_SIZE = (4, 4, 4) 
NUM_SCALES = 2
THRESHOLDS = [0.5]
LINE_COLOR = (0, 255, 255) 
LINE_THICKNESS = 1
OUTPUT_PATH = "" 

def parse_args():
    parser = argparse.ArgumentParser(description='Check 3D Data Visualization with Time Dimension (Time as Channel)')
    parser.add_argument('--input', type=str, default="", help='Input volume path')
    parser.add_argument('--vis_type', type=str, default='entropy')
    parser.add_argument('--method', type=str, default='std')
    parser.add_argument('--aggregate', type=str, default='mean')
    parser.add_argument('--image_size', type=int, nargs='+', default=list(IMAGE_SIZE))
    parser.add_argument('--patch_size', type=int, nargs='+', default=list(BASE_PATCH_SIZE))
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
    parser.add_argument('--line_color', type=int, nargs=3, default=list(LINE_COLOR))
    parser.add_argument('--line_thickness', type=int, default=LINE_THICKNESS)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for GIF')
    parser.add_argument('--no_resize', action='store_true')
    parser.add_argument('--bg_threshold', type=float, default=1e-3, help='Threshold to remove empty patches')
    return parser.parse_args()

def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 1: return (x[0], x[0], x[0])
        return tuple(x)
    return (x, x, x)

def load_volume(path):
    print(f"Loading: {path}")
    if path.endswith('.npy'):
        data = np.load(path)
    elif path.endswith('.npz'):
        with np.load(path) as f:
            keys = list(f.keys())
            if 'arr_0' in keys: data = f['arr_0']
            elif 'arr' in keys: data = f['arr']
            else: data = f[keys[0]]
    else:
        raise ValueError("Unsupported format")
    
    tensor = torch.from_numpy(data).float()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.permute(3, 0, 1, 2) 
            
    return tensor

def force_scale_to_255(img_np):
    img_np = np.nan_to_num(img_np)
    v_min = img_np.min()
    v_max = img_np.max()
    if v_max - v_min < 1e-6:
        return np.zeros_like(img_np, dtype=np.uint8)
    img_norm = (img_np - v_min) / (v_max - v_min)
    return (img_norm * 255).astype(np.uint8)

def get_ortho_frame(vol_3d_numpy, masks_3d_dict, patch_sizes, color, thickness):
    D, H, W = vol_3d_numpy.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
    
    slice_axial = vol_3d_numpy[mid_d, :, :]      
    slice_coronal = vol_3d_numpy[:, mid_h, :]    
    slice_sagittal = vol_3d_numpy[:, :, mid_w]   

    img_ax = cv2.cvtColor(force_scale_to_255(slice_axial), cv2.COLOR_GRAY2RGB)
    img_cor = cv2.cvtColor(force_scale_to_255(slice_coronal), cv2.COLOR_GRAY2RGB)
    img_sag = cv2.cvtColor(force_scale_to_255(slice_sagittal), cv2.COLOR_GRAY2RGB)

    overlay_ax = img_ax.copy()
    overlay_cor = img_cor.copy()
    overlay_sag = img_sag.copy()

    draw_color = color 

    for ps in patch_sizes:
        pd, ph, pw = ps
        if ps not in masks_3d_dict: continue
        mask = masks_3d_dict[ps] 
        if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
        
        if mask.ndim == 4: mask = mask[0]
        
        indices = np.argwhere(mask > 0)
        
        for idx in indices:
            di, hi, wi = idx[0], idx[1], idx[2]
            d_s, d_e = di*pd, min((di+1)*pd, D)
            h_s, h_e = hi*ph, min((hi+1)*ph, H)
            w_s, w_e = wi*pw, min((wi+1)*pw, W)
            
            if d_s <= mid_d < d_e:
                cv2.rectangle(overlay_ax, (w_s, h_s), (w_e, h_e), draw_color, thickness)
            if h_s <= mid_h < h_e:
                cv2.rectangle(overlay_cor, (w_s, d_s), (w_e, d_e), draw_color, thickness)
            if w_s <= mid_w < w_e:
                cv2.rectangle(overlay_sag, (h_s, d_s), (h_e, d_e), draw_color, thickness)

    alpha = 0.4  
    cv2.addWeighted(overlay_ax, alpha, img_ax, 1 - alpha, 0, img_ax)
    cv2.addWeighted(overlay_cor, alpha, img_cor, 1 - alpha, 0, img_cor)
    cv2.addWeighted(overlay_sag, alpha, img_sag, 1 - alpha, 0, img_sag)

    pil_ax = Image.fromarray(img_ax)
    pil_cor = Image.fromarray(img_cor)
    pil_sag = Image.fromarray(img_sag)
    
    target_h = 300
    def resize_h(img, h):
        ratio = h / img.height
        return img.resize((int(img.width * ratio), h), Image.Resampling.NEAREST)

    pil_ax = resize_h(pil_ax, target_h)
    pil_cor = resize_h(pil_cor, target_h)
    pil_sag = resize_h(pil_sag, target_h)
    
    total_w = pil_ax.width + pil_cor.width + pil_sag.width
    combined = Image.new('RGB', (total_w, target_h), (0,0,0))
    combined.paste(pil_ax, (0, 0))
    combined.paste(pil_cor, (pil_ax.width, 0))
    combined.paste(pil_sag, (pil_ax.width + pil_cor.width, 0))
    
    return combined

def filter_background_patches_global(vol_tensor_all, masks_dict, threshold=1e-3):

    print(f"    [Filter] Background filtering (Global max over T > {threshold})...")
    
    filtered_masks = {}
    vol_abs = vol_tensor_all.abs() # (1, T, D, H, W)

    for patch_size, mask in masks_dict.items():
        # Mask shape expected: (1, Nd, Nh, Nw)
        
        # 1. Spatial Max Pool
        # Input: (1, T, D, H, W) -> Output: (1, T, Nd, Nh, Nw)
        patch_max_spatial = F.max_pool3d(vol_abs, kernel_size=patch_size, stride=patch_size)
        
        # 2. Temporal Max (Check if ANY frame has content)
        # Output: (1, 1, Nd, Nh, Nw) -> Squeeze -> (1, Nd, Nh, Nw)
        patch_max_vals, _ = torch.max(patch_max_spatial, dim=1, keepdim=True)
        patch_max_vals = patch_max_vals.squeeze(1) 
        
        if mask.device != patch_max_vals.device:
            patch_max_vals = patch_max_vals.to(mask.device)
            
        has_content_mask = patch_max_vals >= threshold
        new_mask = mask.bool() & has_content_mask
        new_mask = new_mask.float()
        
        filtered_masks[patch_size] = new_mask
        
    return filtered_masks

def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3 
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3 
        print(f"    [GPU Memory | {tag}] Alloc: {allocated:.2f} GB | Max Alloc: {max_allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        print(f"    [GPU Memory | {tag}] Running on CPU")

def process_volume(args):
    vol_tensor_raw = load_volume(args.input) # (T, D, H, W)
    T, D, H, W = vol_tensor_raw.shape
    
    print(f"  [Info] Total Frames: {T}. Mode: Time as Channel (Process once).")
    print(f"  [Info] Computation Method: {args.method}") 

    input_tensor = vol_tensor_raw.unsqueeze(0) 

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        torch.cuda.reset_peak_memory_stats()

    target_size = to_3tuple(args.image_size)
    base_patch_size = to_3tuple(args.patch_size)

    if not args.no_resize:
        print(f"  [Info] Resizing input tensor {input_tensor.shape}...")
        input_tensor = F.interpolate(input_tensor, size=target_size, mode='trilinear', align_corners=False)
        print(f"  [Info] Resized shape: {input_tensor.shape}")

    log_gpu_memory(tag="Before Compute")
    
    with torch.no_grad():
        if args.method == 'entropy':
            importance_maps = compute_patch_entropy_batched_3d(input_tensor, base_patch_size, args.num_scales)
        elif args.method == 'laplacian':
            importance_maps = compute_patch_laplacian_batched_3d(input_tensor, base_patch_size, args.num_scales, aggregate=args.aggregate)
        elif args.method == 'mse' or args.method == 'upsample_mse':
            importance_maps = compute_patch_mse_batched_3d(input_tensor, base_patch_size, args.num_scales, aggregate=args.aggregate)
        elif args.method == 'std':
            importance_maps = compute_patch_local_std_batched_3d(input_tensor, base_patch_size, args.num_scales)
        else:
            print(f"    [Warning] Unknown method '{args.method}', defaulting to entropy.")
            importance_maps = compute_patch_entropy_batched_3d(input_tensor, base_patch_size, args.num_scales)

        log_gpu_memory(tag=f"After {args.method}")

        batch_masks = select_patches_by_threshold_3d(importance_maps, args.thresholds)
        
        batch_masks = filter_background_patches_global(input_tensor, batch_masks, threshold=args.bg_threshold)
        
        final_masks_cpu = {}
        for k, v in batch_masks.items():
            mask_cpu = v.cpu() # (1, Nd, Nh, Nw)
            final_masks_cpu[k] = mask_cpu.expand(T, *mask_cpu.shape[1:])
        
        final_vol_numpy = input_tensor.squeeze(0).cpu().numpy() # (T, D, H, W)

        del input_tensor, importance_maps, batch_masks
        torch.cuda.empty_cache()
    
    patch_sizes_list = []
    for i in range(args.num_scales):
        scale = 2**i
        patch_sizes_list.append((base_patch_size[0]*scale, base_patch_size[1]*scale, base_patch_size[2]*scale))
        
    frames = []
    print(f"  [Info] Drawing frames (Static Mask applied to all frames)...")
    
    real_T = final_vol_numpy.shape[0]

    for t in range(real_T):
        if t % 10 == 0: print(f"    Drawing frame {t}/{real_T}")
        
        current_frame_masks = {}
        for ps in patch_sizes_list:
            if ps in final_masks_cpu:
                current_frame_masks[ps] = final_masks_cpu[ps][t] 
        
        frame_img = get_ortho_frame(
            vol_3d_numpy=final_vol_numpy[t],
            masks_3d_dict=current_frame_masks,
            patch_sizes=patch_sizes_list,
            color=tuple(args.line_color),
            thickness=args.line_thickness
        )
        frames.append(frame_img)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if len(frames) == 1:
        save_path = args.output if args.output.endswith('.png') else args.output.replace('.gif', '.png')
        frames[0].save(save_path)
        print(f"Saved Single Frame to: {save_path}")
    else:
        save_path = args.output if args.output.endswith('.gif') else args.output + '.gif'
        print(f"  [Info] Saving GIF with {len(frames)} frames...")
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=1000 // args.fps, 
            loop=0
        )
        print(f"Saved GIF to: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    if len(args.thresholds) != args.num_scales - 1:
        args.thresholds = THRESHOLDS
    process_volume(args)

# ---------------------------------------------------------------
# References:
# APT: https://github.com/rccchoudhury/apt
# ---------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict
from torchvision.transforms import functional as TF
import math
import cv2
from PIL import Image
import nibabel as nib
import os

def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)

def compute_patch_local_std_batched_3d(images, patch_size=16, num_scales=2, use_std=True, pad_mode='replicate'):
    """
    Compute Local Standard Deviation (or Variance) maps for 3D volumes.
    Optimized for MRI/CT structural complexity analysis.
    
    Args:
        images: torch.Tensor of shape (B, C, D, H, W). 
                Ideally C=1 (grayscale/intensity).
        patch_size: base patch size int or tuple (pd, ph, pw) (default: 16)
        num_scales: number of scales to compute (default: 2)
        use_std: If True, return Standard Deviation. If False, return Variance. (default: True)
        pad_mode: Padding mode for boundary handling. 'replicate' is recommended for MRI 
                  to avoid edge artifacts. Options: 'constant', 'replicate', 'reflect'.
    
    Returns:
        batch_score_maps: dict mapping patch_size (tuple) -> score_map (B, D_p, H_p, W_p)
    """
    batch_size, channels, D, H, W = images.shape
    device = images.device
    base_patch_size = to_3tuple(patch_size)
    
    # 1. Pre-computation for Integral Image trick (E[X^2] - E[X]^2)
    # Using the raw images directly. 
    # Ensure float32 for numerical stability during squaring.
    img_float = images.float()

    img_sq = img_float.pow(2)
    
    batch_score_maps = {}
    
    # Generate scales: e.g., 16, 32, 64
    patch_sizes_list = []
    for i in range(num_scales):
        scale = 2**i
        patch_sizes_list.append((
            base_patch_size[0] * scale,
            base_patch_size[1] * scale,
            base_patch_size[2] * scale
        ))
        
    for ps in patch_sizes_list:
        pd, ph, pw = ps
        
        # 2. Calculate Output Grid Size (Same logic as your Entropy code)
        num_patches_d = (D + pd - 1) // pd
        num_patches_h = (H + ph - 1) // ph
        num_patches_w = (W + pw - 1) // pw
        
        # 3. Calculate Padding
        # This ensures strict alignment with the output shape of your previous functions
        pad_d = num_patches_d * pd - D
        pad_h = num_patches_h * ph - H
        pad_w = num_patches_w * pw - W
        
        # F.pad order: (Left, Right, Top, Bottom, Front, Back) -> (W, H, D)
        # We perform padding *before* pooling to handle the boundaries correctly
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_tuple = (0, pad_w, 0, pad_h, 0, pad_d)
            # Use replicate to prevent "black border" high variance artifacts in MRI
            padded_img = F.pad(img_float, pad_tuple, mode=pad_mode)
            padded_img_sq = F.pad(img_sq, pad_tuple, mode=pad_mode)
        else:
            padded_img = img_float
            padded_img_sq = img_sq
            
        # 4. Compute Local Stats using AvgPool3d (Vectorized & Fast)
        # This replaces the heavy "unfold" operation.
        # Kernel = PatchSize, Stride = PatchSize -> Non-overlapping windows
        
        # E[X]
        mean_map = F.avg_pool3d(padded_img, kernel_size=ps, stride=ps)
        
        # E[X^2]
        mean_sq_map = F.avg_pool3d(padded_img_sq, kernel_size=ps, stride=ps)
        
        # Var(X) = E[X^2] - (E[X])^2
        variance_map = mean_sq_map - mean_map.pow(2)
        
        # Numerical stability: clamp negative values caused by floating point errors
        variance_map = torch.clamp(variance_map, min=0)
        
        # 5. Result Shaping
        # Output shape: (B, C, Nt, Nh, Nw) -> (B, Nt, Nh, Nw) if C=1
        # If input has multiple channels, we usually average the variance across channels
        # or keep it. Here we average C to get a single complexity map per location.
        if channels > 1:
            variance_map = variance_map.mean(dim=1)
        else:
            variance_map = variance_map.squeeze(1)
            
        if use_std:
            score_map = torch.sqrt(variance_map + 1e-6)
        else:
            score_map = variance_map
            
        batch_score_maps[ps] = score_map
        
        
    return batch_score_maps


def compute_shared_random_score_maps(images, patch_size=16, num_scales=2, keep_ratio=0.5, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
        
    batch_size, channels, D, H, W = images.shape
    device = images.device
    base_patch_size = to_3tuple(patch_size)
    
    batch_score_maps = {}
    
    patch_sizes_list = []
    for i in range(num_scales):
        scale = 2**i
        patch_sizes_list.append((
            base_patch_size[0] * scale,
            base_patch_size[1] * scale,
            base_patch_size[2] * scale
        ))

    for ps in patch_sizes_list:
        pd, ph, pw = ps
        
        num_patches_d = (D + pd - 1) // pd
        num_patches_h = (H + ph - 1) // ph
        num_patches_w = (W + pw - 1) // pw
        

        shared_rand_tensor = torch.rand((1, num_patches_d, num_patches_h, num_patches_w), device=device)
        
        shared_score_map = (shared_rand_tensor < keep_ratio).float()

        score_map = shared_score_map.expand(batch_size, -1, -1, -1)
        
        batch_score_maps[ps] = score_map
        
    return batch_score_maps

def compute_patch_local_attention_batched_3d(images, patch_size=16, num_scales=2, 
                                         guidance_map="", guidance_thresholds=[0.0,0.0]):

    batch_size, channels, D, H, W = images.shape
    base_patch_size = to_3tuple(patch_size)
    device = images.device
    
    batch_score_maps = {}
    
    if guidance_map is None:
        raise ValueError("Guidance Map is required (Tensor or Path to .nii.gz).")

    if isinstance(guidance_map, str):
        if not os.path.exists(guidance_map):
            raise FileNotFoundError(f"Guidance map file not found: {guidance_map}")
        
        nii_obj = nib.load(guidance_map)
        nii_data = nii_obj.get_fdata()
        
        guidance_map = torch.from_numpy(nii_data).float().to(device)
    if isinstance(guidance_map, torch.Tensor) and guidance_map.device != device:
        guidance_map = guidance_map.to(device)

    if guidance_map.ndim == 3:
        guidance_map = guidance_map.unsqueeze(0).unsqueeze(0) # (1, 1, D', H', W')
    elif guidance_map.ndim == 4:
        guidance_map = guidance_map.unsqueeze(0) 
        

    # (1, 1, D, H, W) -> (B, 1, D, H, W)
    if guidance_map.shape[0] != batch_size:
        guidance_map = guidance_map.expand(batch_size, -1, -1, -1, -1)

    if guidance_thresholds is None:
        guidance_thresholds = [0.0] * num_scales

    patch_sizes_list = []
    for i in range(num_scales):
        scale = 2**i
        patch_sizes_list.append((
            base_patch_size[0] * scale,
            base_patch_size[1] * scale,
            base_patch_size[2] * scale
        ))
    
    for i, ps in enumerate(patch_sizes_list):
        pd, ph, pw = ps
        
        num_patches_d = (D + pd - 1) // pd
        num_patches_h = (H + ph - 1) // ph
        num_patches_w = (W + pw - 1) // pw
        
        pad_d = num_patches_d * pd - D
        pad_h = num_patches_h * ph - H
        pad_w = num_patches_w * pw - W
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_tuple = (0, pad_w, 0, pad_h, 0, pad_d)
            padded_guidance = F.pad(guidance_map, pad_tuple, mode='constant', value=0)
        else:
            padded_guidance = guidance_map

        guidance_grid = F.max_pool3d(padded_guidance, kernel_size=ps, stride=ps)
        
        # (B, 1, Dp, Hp, Wp) -> (B, Dp, Hp, Wp)
        if guidance_grid.shape[1] == 1:
            score_map = guidance_grid.squeeze(1) 
        else:
            score_map = guidance_grid.mean(dim=1)
            
        thresh = guidance_thresholds[i] if i < len(guidance_thresholds) else 0.0
        
        mask = (score_map >= thresh).float()
        score_map = score_map * mask 
            
        batch_score_maps[ps] = score_map
        
    return batch_score_maps


def compute_patch_entropy_batched_3d(images, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    """
    Compute entropy maps for multiple patch sizes in a batch of 3D volumes (Video/Medical)
    using vectorized operations.
    
    Args:
        images: torch.Tensor of shape (B, C, D, H, W) with values in range [0, 1]
        patch_size: base patch size int or tuple (pt, ph, pw) (default: 16)
        num_scales: number of scales to compute (default: 2)
        bins: number of bins for histogram (default: 512)
        pad_value: high entropy value to pad incomplete patches with (default: 1e6)
    
    Returns:
        batch_entropy_maps: dict mapping patch_size (tuple) -> entropy_map (B, D_p, H_p, W_p)
    """
    batch_size, channels, D, H, W = images.shape
    device = images.device
    base_patch_size = to_3tuple(patch_size)
    
    grayscale_images = images.mean(dim=1)
    
    batch_entropy_maps = {}
    
    # Generate scales. Assuming isotropic scaling for simplicity, or scale all dims
    patch_sizes_list = []
    for i in range(num_scales):
        scale = 2**i
        patch_sizes_list.append((
            base_patch_size[0] * scale,
            base_patch_size[1] * scale,
            base_patch_size[2] * scale
        ))
    
    for ps in patch_sizes_list:
        pd, ph, pw = ps
        
        num_patches_d = (D + pd - 1) // pd
        num_patches_h = (H + ph - 1) // ph
        num_patches_w = (W + pw - 1) // pw
        
        # Calculate Padding (Left, Right, Top, Bottom, Front, Back)
        pad_d = num_patches_d * pd - D
        pad_h = num_patches_h * ph - H
        pad_w = num_patches_w * pw - W
        
        padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        
        patches = padded_images.unfold(1, pd, pd).unfold(2, ph, ph).unfold(3, pw, pw)
        
        # Total pixels per patch
        pixels_per_patch = pd * ph * pw
        
        flat_patches = patches.reshape(batch_size, num_patches_d, num_patches_h, num_patches_w, pixels_per_patch)
        
        # Quantize
        flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins-1)
        
        # Vectorized Histogram
        reshaped_patches = flat_patches_int.reshape(-1, pixels_per_patch)
        
        one_hot = torch.zeros(reshaped_patches.size(0), pixels_per_patch, bins, device=device)
        one_hot = one_hot.scatter_(2, reshaped_patches.unsqueeze(2), 1)
        
        histograms = one_hot.sum(1) # Sum over pixels
        histograms = histograms.reshape(batch_size, num_patches_d, num_patches_h, num_patches_w, bins)
        
        # Probability & Entropy
        probabilities = histograms.float() / pixels_per_patch
        epsilon = 1e-10
        entropy_map = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=4)
        
        # Handle Padding Values
        if pad_d > 0: entropy_map[:, -1, :, :] = pad_value
        if pad_h > 0: entropy_map[:, :, -1, :] = pad_value
        if pad_w > 0: entropy_map[:, :, :, -1] = pad_value
        
        # Store using the tuple as key, or just the height (ps[1]) if you prefer integer keys
        batch_entropy_maps[ps] = entropy_map
    
    return batch_entropy_maps

def select_patches_by_threshold_3d(entropy_maps, thresholds, images=None, bg_threshold=1e-2):

    patch_sizes = sorted(list(entropy_maps.keys()), key=lambda x: x[0]*x[1]*x[2])
    
    masks = {}
    
    def get_foreground_mask(p_size):
        if images is None:
            return torch.ones_like(entropy_maps[p_size])
            
        pd, ph, pw = p_size
        
        D, H, W = images.shape[2:]
        pad_d = (pd - D % pd) % pd
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        
        padded_img = F.pad(images, (0, pad_w, 0, pad_h, 0, pad_d), value=0)
        
        max_pool = F.max_pool3d(padded_img.abs(), kernel_size=p_size, stride=p_size)
        
        if max_pool.shape[1] > 1:
            max_pool = max_pool.amax(dim=1) # (B, Dg, Hg, Wg)
        else:
            max_pool = max_pool.squeeze(1)
            
        return (max_pool > bg_threshold).float()

    base_mask = get_foreground_mask(patch_sizes[0])
    masks[patch_sizes[0]] = base_mask
    

    if images is not None:
        total_voxels = base_mask.numel()
        fg_voxels = base_mask.sum().item()

        # bg_ratio = 1.0 - (fg_voxels / total_voxels)
        # print(f"[Info] Base Scale {patch_sizes[0]}: Background Ratio = {bg_ratio:.2%} "
        #       f"(BG/Total: {int(total_voxels - fg_voxels)}/{total_voxels})")
    
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        threshold = thresholds[i-1]
        
        fg_mask = get_foreground_mask(current_size)
        
        score_condition = (entropy_maps[current_size] < threshold).float()
        masks[current_size] = score_condition * fg_mask
        
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        for j in range(i):
            smaller_size = patch_sizes[j]
            
            s_d = current_size[0] // smaller_size[0]
            s_h = current_size[1] // smaller_size[1]
            s_w = current_size[2] // smaller_size[2]
            
            mask_upscaled = masks[current_size].repeat_interleave(s_d, dim=1)\
                                               .repeat_interleave(s_h, dim=2)\
                                               .repeat_interleave(s_w, dim=3)
            
            D_s, H_s, W_s = entropy_maps[smaller_size].shape[1:]
            mask_upscaled = mask_upscaled[:, :D_s, :H_s, :W_s]
            
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
            
    return masks

def compute_patch_laplacian_batched_3d(images, patch_size=16, num_scales=2, aggregate='mean', pad_mode='reflect'):

    if images.ndim != 5:
        raise ValueError(f"Input must be 5D (B, T, D, H, W), got {images.shape}")
        
    B, T, D, H, W = images.shape
    device = images.device
    base_patch_size = to_3tuple(patch_size) # (pd, ph, pw)

    # Shape: (B, T, D, H, W) -> (B, D, H, W)
    mean_images = images.mean(dim=1)
    
    inp = mean_images.unsqueeze(1) # (B, 1, D, H, W)

    k = torch.ones((3, 3, 3), device=device)
    k[1, 1, 1] = -26.0
    laplacian_kernel = k.view(1, 1, 3, 3, 3) # (Out, In, kD, kH, kW)
    
    padded_inp = F.pad(inp, (1, 1, 1, 1, 1, 1), mode=pad_mode)
    
    resp = F.conv3d(padded_inp, laplacian_kernel)
    resp = torch.abs(resp.squeeze(1)) 
    
    batch_laplacian_maps = {}
    
    patch_sizes_list = []
    for i in range(num_scales):
        scale = 2**i
        patch_sizes_list.append((
            base_patch_size[0] * scale,
            base_patch_size[1] * scale,
            base_patch_size[2] * scale
        ))
        
    for ps in patch_sizes_list:
        pd, ph, pw = ps # Patch depth, height, width
        

        nd = (D + pd - 1) // pd
        nh = (H + ph - 1) // ph
        nw = (W + pw - 1) // pw
        
        pad_d = nd * pd - D
        pad_h = nh * ph - H
        pad_w = nw * pw - W
        
        padded_resp = F.pad(resp, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
        
        # Input: (B, D_pad, H_pad, W_pad)
        # Dimension 1 is D, 2 is H, 3 is W
        patches = padded_resp.unfold(1, pd, pd).unfold(2, ph, ph).unfold(3, pw, pw)
        # Output Shape: (B, Nd, Nh, Nw, pd, ph, pw)
        
        dims_to_agg = (4, 5, 6) 
        
        if aggregate == 'mean':
            patch_val = patches.mean(dim=dims_to_agg)
        elif aggregate == 'max':
            patch_val = patches.amax(dim=dims_to_agg)
        elif aggregate == 'std':
            patch_val = patches.std(dim=dims_to_agg)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")
            
        batch_laplacian_maps[ps] = patch_val

    return batch_laplacian_maps

def compute_patch_mse_batched_3d(images, patch_size=16, num_scales=3, scale_factors=None, aggregate='mean'):
    """
    3D MSE Computation via Trilinear Interpolation.
    """
    batch_size, channels, D, H, W = images.shape
    base_patch_size = to_3tuple(patch_size)
    
    if images.max() > 1.0:
        normalized_images = images / 255.0
    else:
        normalized_images = images.clone()
        
    if scale_factors is None:
        scale_factors = [0.5, 0.5, 0.25]
    if len(scale_factors) < num_scales:
        scale_factors.extend([scale_factors[-1]] * (num_scales - len(scale_factors)))
        
    patch_sizes_list = []
    for i in range(num_scales):
        scale = 2**i
        patch_sizes_list.append((
            base_patch_size[0] * scale,
            base_patch_size[1] * scale,
            base_patch_size[2] * scale
        ))
        
    batch_mse_maps = {}
    
    for i, (ps, sf) in enumerate(zip(patch_sizes_list, scale_factors)):
        # 3D Interpolation
        img_down = F.interpolate(normalized_images, scale_factor=sf, mode='trilinear', align_corners=False)
        img_up = F.interpolate(img_down, scale_factor=1.0/sf, mode='trilinear', align_corners=False)
        
        if img_up.shape[2:] != (D, H, W):
            img_up = F.interpolate(img_up, size=(D, H, W), mode='trilinear', align_corners=False)
            
        mse_per_voxel = F.mse_loss(normalized_images, img_up, reduction='none')
        mse_map = mse_per_voxel.mean(dim=1) # Average channels -> (B, T, H, W)
        
        pd, ph, pw = ps
        nt = (D + pd - 1) // pd
        nh = (H + ph - 1) // ph
        nw = (W + pw - 1) // pw
        
        pad_t, pad_h, pad_w = nt*pd - D, nh*ph - H, nw*pw - W
        padded_mse = F.pad(mse_map, (0, pad_w, 0, pad_h, 0, pad_t), value=0)
        
        patches = padded_mse.unfold(1, pd, pd).unfold(2, ph, ph).unfold(3, pw, pw)
        
        dims_to_agg = (4, 5, 6)
        if aggregate == 'mean': val = patches.mean(dim=dims_to_agg)
        elif aggregate == 'max': val = patches.amax(dim=dims_to_agg)
        elif aggregate == 'std': val = patches.std(dim=dims_to_agg)
        
        batch_mse_maps[ps] = val
        
    return batch_mse_maps

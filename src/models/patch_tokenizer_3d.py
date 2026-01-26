# ---------------------------------------------------------------
# References:
# APT: https://github.com/rccchoudhury/apt
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Dict, List, Tuple, Union, Optional


from .entropy_utils_3d import (
    select_patches_by_threshold_3d,
    compute_patch_entropy_batched_3d,
    compute_patch_local_std_batched_3d,
    compute_patch_local_attention_batched_3d,
    compute_shared_random_score_maps,
    compute_patch_mse_batched_3d,
    compute_patch_laplacian_batched_3d
)

def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)

class PatchTokenizer3D(nn.Module):
    def __init__(
        self,
        num_scales: int,
        base_patch_size: Union[int, Tuple[int, int, int]],
        image_size: Union[int, Tuple[int, int, int]],
        thresholds: List[float],
        mean: List[float] = [1.0],
        std: List[float] = [0.0],
        method: str = 'std',
    ):
        super().__init__()
        self.num_scales = num_scales
        self.base_patch_size = to_3tuple(base_patch_size)
        self.image_size = to_3tuple(image_size)
        self.thresholds = thresholds
        self.method = method

        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1, 1))

    def _get_current_patch_size(self, scale_idx):
        scale = 2 ** scale_idx
        return (
            self.base_patch_size[0] * scale,
            self.base_patch_size[1] * scale,
            self.base_patch_size[2] * scale
        )

    def compute_importance_maps(self, images: torch.Tensor) -> Dict:
        if self.method == 'entropy':
            return compute_patch_entropy_batched_3d(images, self.base_patch_size, self.num_scales)
        elif self.method == 'std':
            return compute_patch_local_std_batched_3d(images, self.base_patch_size, self.num_scales)
        elif self.method == 'attention':
            return compute_patch_local_attention_batched_3d(images, self.base_patch_size, self.num_scales)
        elif self.method == 'random':
            return compute_shared_random_score_maps(images, self.base_patch_size, self.num_scales)
        elif self.method == 'mse':
            return compute_patch_mse_batched_3d(images, self.base_patch_size, self.num_scales)
        elif self.method == 'laplacian':
            return compute_patch_laplacian_batched_3d(images, self.base_patch_size, self.num_scales)
        else:
            raise ValueError(f"Method {self.method} unknown")


    def forward(self, images):
        B, C, D, H, W = images.shape
        device = images.device
        
        imp_maps = self.compute_importance_maps(images)
        
        masks = select_patches_by_threshold_3d(imp_maps, thresholds=self.thresholds, images=images)
        
        output_dict = {}
        batch_scales_list = [[] for _ in range(B)]
        
        seqlens = torch.zeros(B, dtype=torch.long, device=device)

        for idx in range(self.num_scales):
            cur_patch_size = self._get_current_patch_size(idx)
            key = cur_patch_size
            
            # Mask: (B, Dg, Hg, Wg)
            mask = masks[key]
            Dg, Hg, Wg = mask.shape[1:]
            
            if idx == 0:
                scale_img = images
            else:
                scale_img = F.interpolate(images, scale_factor=0.5**idx, mode='trilinear', align_corners=False)
            
            pd, ph, pw = self.base_patch_size 
            
            # Einops rearrange
            patches_grid = einops.rearrange(
                scale_img, 
                'b c (d pd) (h ph) (w pw) -> b d h w c pd ph pw', 
                pd=pd, ph=ph, pw=pw
            )
            
            mask_bool = mask.bool() # (B, Dg, Hg, Wg)
            selected_patches = patches_grid[mask_bool] # Packed Tensor
            
            output_dict[f"resized_patches_{key}"] = selected_patches
            output_dict[f"pos_embed_mask_{key}"] = mask_bool.flatten(1) 
            
            if idx > 0:
                pdl, phl, pwl = cur_patch_size
                patches_grid_full = einops.rearrange(
                    images,
                    'b c (d pd) (h ph) (w pw) -> b d h w c pd ph pw',
                    pd=pdl, ph=phl, pw=pwl
                )
                output_dict[f"full_patches_{key}"] = patches_grid_full[mask_bool]
            else:
                output_dict[f"full_patches_{key}"] = selected_patches
            

            coord_grid = self._generate_coordinate_grid((Dg, Hg, Wg), cur_patch_size, device)
            
            
            # mask_bool: (B, Dg, Hg, Wg) -> flatten -> (B, N_grid)
            flat_mask = mask_bool.flatten(1)
            counts = flat_mask.sum(1) # (B,)
            seqlens += counts
            
            
            for b in range(B):
                # mask[b]: (Dg, Hg, Wg)
                # coord_grid: (Dg, Hg, Wg, 3)
                m = mask_bool[b]
                if m.any():
                    # select coordinates
                    c = coord_grid[m] # (n_patches, 3)             
                    # scale index
                    s = torch.full((c.shape[0],), idx, dtype=torch.long, device=device)
                    batch_scales_list[b].append(s)

        final_scales_list = [torch.cat(x, dim=0) if x else torch.empty(0, dtype=torch.long, device=device) for x in batch_scales_list]
        
        # Pad sequence: (B, N_max)
        padded_scales = torch.nn.utils.rnn.pad_sequence(final_scales_list, batch_first=True, padding_value=-1) # -1 为 padding

        output_mask = padded_scales + 1
        cls_col = torch.full((B, 1), -1, device=device, dtype=output_mask.dtype)
        output_mask = torch.cat([cls_col, output_mask], dim=1)

        seqlens_with_cls = seqlens + 1
        output_dict['patch_scale_indices'] = padded_scales # (B, N_max)
        output_dict['output_mask'] = output_mask

        output_dict['seqlens'] = seqlens_with_cls.tolist()

        output_dict["cu_seqlens"] = torch.tensor([0] + list(seqlens_with_cls.cumsum(0).cpu()), dtype=torch.int32, device=device)
        output_dict["max_seqlen"] = seqlens_with_cls.max().item()

        return output_dict

from typing import Union, Tuple, List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from timm.layers.format import Format
from timm.layers.trace_utils import _assert

_logger = logging.getLogger(__name__)

def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)

def resample_patch_embed(
    patch_embed_weight: torch.Tensor,
    new_size: Union[int, Tuple[int, int, int]],
    verbose: bool = False
) -> torch.Tensor:
    old_size = patch_embed_weight.shape[2:]
    new_size = to_3tuple(new_size)
    if old_size == new_size:
        return patch_embed_weight

    if verbose:
        _logger.info(f"Resizing 3D patch embedding weights from {old_size} to {new_size}")

    weight = F.interpolate(
        patch_embed_weight, 
        size=new_size, 
        mode='trilinear', 
        align_corners=False
    )
    return weight

def resample_abs_pos_embed_3d(
    posemb,
    new_size: Tuple[int, int, int],
    old_size: Tuple[int, int, int],
    num_prefix_tokens: int = 1,
    verbose: bool = False,
):
    if new_size == old_size:
        return posemb

    B = posemb.shape[0]

    if num_prefix_tokens:
        prefix_tokens = posemb[:, :num_prefix_tokens]
        posemb = posemb[:, num_prefix_tokens:]
    else:
        prefix_tokens = None

    embed_dim = posemb.shape[-1]
    # Reshape logic assumes flattened (D, H, W)
    # posemb = posemb.reshape(1, old_size[0], old_size[1], old_size[2], embed_dim).permute(0, 4, 1, 2, 3)
    posemb = posemb.reshape(B, old_size[0], old_size[1], old_size[2], embed_dim).permute(0, 4, 1, 2, 3)

    origin_dtype = posemb.dtype
    posemb = posemb.float() 

    posemb = F.interpolate(
        posemb, 
        size=new_size, 
        mode='trilinear', 
        align_corners=False
    )

    posemb = posemb.to(origin_dtype)

    posemb = posemb.permute(0, 2, 3, 4, 1).flatten(1, 3)

    if prefix_tokens is not None:
        posemb = torch.cat([prefix_tokens, posemb], dim=1)

    if verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb

# --- 3D Patch Embedding ---

class PatchEmbed3D(nn.Module):
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[Tuple[int, int, int]] = (96, 96, 96),
            patch_size: Tuple[int, int, int] = (4, 4, 4),
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if img_size is not None:
            self.img_size = to_3tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.embed_dim = embed_dim
        self.flatten = flatten
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def set_input_size(
                self,
                img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
                patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        ):
            new_patch_size = None
            if patch_size is not None:
                new_patch_size = to_3tuple(patch_size)
                
            if new_patch_size is not None and new_patch_size != self.patch_size:
                with torch.no_grad():
                    new_proj = nn.Conv3d(
                        self.proj.in_channels,
                        self.proj.out_channels,
                        kernel_size=new_patch_size,
                        stride=new_patch_size, 
                        bias=self.proj.bias is not None,
                    )
                    new_weight = resample_patch_embed(
                        self.proj.weight, 
                        new_patch_size, 
                        verbose=True
                    )
                    new_proj.weight.copy_(new_weight)
                    if self.proj.bias is not None:
                        new_proj.bias.copy_(self.proj.bias)
                    self.proj = new_proj
                self.patch_size = new_patch_size

            img_size = img_size or self.img_size
            if img_size is not None:
                img_size = to_3tuple(img_size)

            if img_size != self.img_size or new_patch_size is not None:
                self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        if img_size is None:
            return None, None, None
        img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.strict_img_size and self.img_size is not None:
            _assert(D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], 
                    f"Input dimensions {x.shape[2:]} do not match model requirements {self.img_size}")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    def forward_patch(self, patches):
        x = self.proj(patches)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.norm(x)
        return x


# --- 3D Tokenized Zero Conv Patch Attention ---
class TokenizedZeroConvPatchAttn3D(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        embed_dim: Optional[int] = None,
        num_scales: int = 2,
        in_chans: int = 40,
        pos_drop_rate: float = 0.0,
        thresholds: Optional[List[float]] = None):

        super().__init__()
        self.img_size = to_3tuple(image_size)
        self.base_patch_size = to_3tuple(patch_size)
        self.num_scales = num_scales
        
        self.patch_sizes = []
        for i in range(num_scales):
            scale = 2**i
            self.patch_sizes.append(
                (self.base_patch_size[0] * scale, 
                 self.base_patch_size[1] * scale, 
                 self.base_patch_size[2] * scale)
            )
        self.thresholds = thresholds
        self.alpha_schedule = False
        self.embed_dim = embed_dim
        
        t_grid = self.img_size[0] // self.base_patch_size[0]
        h_grid = self.img_size[1] // self.base_patch_size[1]
        w_grid = self.img_size[2] // self.base_patch_size[2]
        self.num_patches = (t_grid * h_grid * w_grid) + 1

        self.patch_embed = PatchEmbed3D(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=in_chans,
            flatten=True
        )

        self.patch_attn = nn.Conv3d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2
        )
        
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        
        self.zero_conv = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) 

    def forward(self, x, base_pos_embed, input_dict, current_img_size=None):
        batch_size = x.shape[0]

        output_mask = input_dict["output_mask"] # Shape (B, N_max) or (Total,)
        
        def _resolve_key(prefix: str, size_tuple: Tuple[int, int, int]):
            k1 = f"{prefix}_{size_tuple}"
            if k1 in input_dict: return k1
            k2 = f"{prefix}_{size_tuple[0]}_{size_tuple[1]}_{size_tuple[2]}"
            if k2 in input_dict: return k2
            raise KeyError(f"Could not resolve key for prefix={prefix} size={size_tuple}")
        base_key = _resolve_key("resized_patches", self.base_patch_size)
        posmask_key = _resolve_key("pos_embed_mask", self.base_patch_size)
        base16 = input_dict[base_key]
        posmask_16 = input_dict[posmask_key]

        # --- Resample Base Pos Embed ---
        current_pos_embed = base_pos_embed

        if current_pos_embed.shape[0] == batch_size:
            pos_embed16 = current_pos_embed[:, 1:] 
        elif current_pos_embed.shape[0] == 1:
            pos_embed16 = current_pos_embed[:, 1:].repeat(batch_size, 1, 1)

        # Handle cases where posmask is flat (packed) or batched
        if  posmask_16.ndim == 2: # (B, N)
             # If posmask is (B, N), we can mask and flatten or keep batched.
             # But base16 is likely PACKED (Total, ...).
             # So we must apply mask to flatten pos_embed16.
            pos_embed16 = pos_embed16[posmask_16.bool()]
        elif current_pos_embed.size(1) == posmask_16.size(1):
            pos_embed16 = current_pos_embed.repeat(batch_size, 1, 1)
            pos_embed16 = pos_embed16[posmask_16.bool()]

        expanded_rope_mask = None

        # CLS Token Pos Embed
        cls_token_pos_embed = current_pos_embed[:, :1] # (1, 1, D)

        # Process Base Scale
        embed16 = self.patch_embed.forward_patch(base16) + pos_embed16
        
        # Check if output_mask is Batched (2D) or Packed (1D)
        is_batched = (output_mask.ndim == 2)
        
        if is_batched:
            B_mask, N_max = output_mask.shape
            expanded_outputs = torch.zeros((B_mask, N_max, self.embed_dim), device=embed16.device, dtype=embed16.dtype)
            
            # Prepare CLS token for broadcasting
            # (1, 1, D) -> (B, D)
            cls_token_val = self.cls_token + cls_token_pos_embed
            cls_token_val = cls_token_val.expand(B_mask, -1, -1).squeeze(1)
            
            # Assign CLS (where mask == -1)
            # Mask (B, N). LHS: expanded_outputs[mask] -> (Count, D)
            # RHS: cls_token_val -> (B, D).
            # This requires Count == B. Which is true for CLS tokens (1 per sample).
            expanded_outputs[output_mask == -1] = cls_token_val
            
            # Assign Base Patches (where mask == 1)
            expanded_outputs[output_mask == 1] = embed16
        else:
            # Legacy Packed Mode
            expanded_outputs = torch.zeros((output_mask.shape[0], self.embed_dim), device=embed16.device, dtype=embed16.dtype)
            expanded_outputs[output_mask == -1] = (self.cls_token + cls_token_pos_embed).squeeze(0)
            expanded_outputs[output_mask == 1] = embed16
            
        cls_tok_loc = None # Not strictly needed for MAE if we use batched output

        # Process larger scales
        for scale_idx, cur_patch_size in enumerate(self.patch_sizes[1:]):
            resized_key = _resolve_key("resized_patches", cur_patch_size)
            full_key = _resolve_key("full_patches", cur_patch_size)
            posmask_key = _resolve_key("pos_embed_mask", cur_patch_size)
            base_patches = input_dict[resized_key]
            full_patches = input_dict[full_key]
            pos_embed_masks = input_dict[posmask_key]
            # ... Pos Embed Resampling logic ...
            # Simplify for this fix: calculate new_grid
            if current_img_size:
                new_grid = tuple(s // p for s, p in zip(current_img_size, cur_patch_size))
            else:
                new_grid = tuple(s // p for s, p in zip(self.img_size, cur_patch_size))
            
            # Assuming base_pos_embed matches base_patch_size grid
            src_grid = tuple(s // p for s, p in zip(self.img_size, self.base_patch_size))
            resampled_pos_embed = resample_abs_pos_embed_3d(
                base_pos_embed,
                new_size=new_grid,
                old_size=src_grid,
                num_prefix_tokens=1
            )
            # pos_embed_slice = resampled_pos_embed[:, 1:] # remove cls token
            pos_embed_slice = resampled_pos_embed[:, 1:].repeat(batch_size, 1, 1)
            pos_embed = pos_embed_slice[pos_embed_masks.bool()]

            if pos_embed_masks.sum() > 0:
                # 1. Base Path
                embed_scale = self.patch_embed.forward_patch(base_patches)
                
                # 2. Detail Path (Aggregating 2x2x2)
                scale_factor = 2 ** (scale_idx + 1)
                in_ch = self.patch_embed.proj.in_channels
                
                if full_patches.ndim == 2:
                    full_patches = full_patches.view(-1, in_ch, *self.base_patch_size)

                # View as small blocks (N, C, 2*Tb, 2*Hb, 2*Wb) -> (N, 8, C, Tb, Hb, Wb)
                # This logic splits the large patch into 8 small ones
                full_patches = full_patches.view(
                    -1, in_ch,
                    scale_factor, self.base_patch_size[0],
                    scale_factor, self.base_patch_size[1],
                    scale_factor, self.base_patch_size[2]
                )
                full_patches = full_patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
                full_patches = full_patches.view(-1, in_ch, *self.base_patch_size)
                
                features = self.patch_embed.forward_patch(full_patches)
                
                features = features.view(-1, scale_factor, scale_factor, scale_factor, self.embed_dim)
                features = features.permute(0, 4, 1, 2, 3) 
                
                for _ in range(scale_idx + 1):
                    features = self.patch_attn(features)
                attn_scale = features.view(-1, self.embed_dim)
                embed_scale = self.zero_conv(attn_scale) + embed_scale + pos_embed
            else:
                 embed_scale = torch.zeros((0, self.embed_dim), device=x.device, dtype=expanded_outputs.dtype)

            # Assign to output
            expanded_outputs[output_mask == (scale_idx+2)] = embed_scale.float()

        if not is_batched:
            expanded_outputs = expanded_outputs.unsqueeze(0)

        expanded_outputs = self.pos_drop(expanded_outputs)
        cu_seqlens = input_dict["cu_seqlens"]
        max_seqlen = input_dict["max_seqlen"]
        
        return expanded_outputs, cu_seqlens, max_seqlen, cls_tok_loc, expanded_rope_mask
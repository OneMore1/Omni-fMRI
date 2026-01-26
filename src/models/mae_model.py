import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer
from .patch_tokenizer_3d import PatchTokenizer3D
from .vit_components import Block


class MAEDecoder(nn.Module):
    def __init__(self, patch_size, img_size, num_classes=1, embed_dim=768, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 gate_attention='elementwise', qk_norm=True, drop_path_rate=0.1, proj_bias=True,
                 num_scales=2): 
        super().__init__()

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.img_size = img_size
        self.patch_size = patch_size # Base patch size (e.g., 4,4,4)
        
        # Grid dimensions for Base Scale
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        
        # Position Embedding Volume
        self.pos_embed_vol = nn.Parameter(
            torch.zeros(1, decoder_embed_dim, *self.grid_size)
        )
        torch.nn.init.trunc_normal_(self.pos_embed_vol, std=.02)

        self.scale_embed = nn.Parameter(torch.zeros(1, num_scales, decoder_embed_dim))
        torch.nn.init.normal_(self.scale_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                qkv_bias=True,
                drop_path=dpr[i],
                gate_attention=gate_attention,
                proj_bias=proj_bias,
                norm_layer=norm_layer
            )
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Multi-Head Predictors
        self.predictors = nn.ModuleList()
        for i in range(num_scales):
            scale_factor = 2 ** i
            current_pixels = (patch_size[0] * scale_factor) * \
                             (patch_size[1] * scale_factor) * \
                             (patch_size[2] * scale_factor) * num_classes
            
            self.predictors.append(nn.Linear(decoder_embed_dim, current_pixels, bias=True))
            print(f"sclaes:{i}, predicted pixel:{current_pixels}")

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_decoder_pos_and_scale_embed_vectorized(self, patch_coords, patch_scale_indices):
        """
        Args:
            patch_coords: (B, N_max, 3) 
            patch_scale_indices: (B, N_max) 
        """
        B, N_max, _ = patch_coords.shape
        D_img, H_img, W_img = self.img_size
        device = patch_coords.device

        safe_indices = patch_scale_indices.clamp(min=0)

        scale_factors = (2 ** safe_indices).unsqueeze(-1) # (B, N, 1)
        patch_sizes = torch.tensor(self.patch_size, device=device).view(1, 1, 3) * scale_factors
        
        centers = patch_coords + patch_sizes / 2.0
        
        # norm_z = 2 * (center_z / D) - 1
        img_size_tensor = torch.tensor([D_img, H_img, W_img], device=device).view(1, 1, 3)
        norm_coords = 2.0 * (centers / img_size_tensor) - 1.0
        
        # (z, y, x) -> (x, y, z) for grid_sample
        grid = norm_coords.flip(-1) # (B, N, 3)
        
        # Reshape for grid_sample: (B, D, H, W, 3) -> Here (B, 1, 1, N, 3)
        grid = grid.view(B, 1, 1, N_max, 3)
        
        # 2. Position Embedding Sampling
        # pos_embed_vol: (1, Dim, Dg, Hg, Wg) -> Expand to B
        pos_vol = self.pos_embed_vol.expand(B, -1, -1, -1, -1)
        
        # Sample: (B, Dim, 1, 1, N)
        sampled_pos = F.grid_sample(
            pos_vol, grid, 
            mode='bilinear', padding_mode='border', align_corners=False
        )
        
        # Reshape back: (B, N, Dim)
        sampled_pos = sampled_pos.squeeze(2).squeeze(2).transpose(1, 2)
        
        # 3. Scale Embedding Gathering
        # self.scale_embed: (1, num_scales, Dim)
        # patch_scale_indices: (B, N)
        valid_indices = patch_scale_indices.clamp(min=0)
        
        scale_embeds = self.scale_embed[0, valid_indices, :] # (B, N, Dim)
        
        return sampled_pos, scale_embeds

    def forward(self, x, ids_restore, input_dict, len_keep=None):
        x = self.decoder_embed(x) # (B, N_vis_max+1, D_dec)

        cls_token = x[:, :1, :]       # (B, 1, D)
        x_patches = x[:, 1:, :]       # (B, N_vis_max, D) 
        
        B, N_vis_max, D = x_patches.shape
        N_max = ids_restore.shape[1]  
    
        x_full_shuffled = self.mask_token.to(dtype=x.dtype).repeat(B, N_max, 1)
        
        if len_keep is not None:
            # mask shape: (B, N_vis_max)
            range_tensor = torch.arange(N_vis_max, device=x.device).unsqueeze(0).expand(B, -1)
            valid_mask = range_tensor < len_keep.unsqueeze(1)
            x_full_shuffled[:, :N_vis_max, :][valid_mask] = x_patches[valid_mask]
        else:
            print("Warning: len_keep not provided to decoder, assuming fixed length.")
            x_full_shuffled = torch.cat([x_patches, self.mask_token.repeat(B, N_max - N_vis_max, 1)], dim=1)

        x_full = torch.gather(x_full_shuffled, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        patch_coords = input_dict['patch_coords']       
        patch_scales = input_dict['patch_scale_indices'] 

        pos_embed, scale_embed = self.get_decoder_pos_and_scale_embed_vectorized(patch_coords, patch_scales)
        
        # Add Embeddings
        x_full = x_full + pos_embed + scale_embed
        
        # Append CLS back
        x = torch.cat([cls_token, x_full], dim=1)

        total_seqlens = torch.as_tensor(input_dict['seqlens'], device=x.device)
        cu_seqlens = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int32), 
                                total_seqlens.cumsum(0, dtype=torch.int32)])
        max_seqlen = total_seqlens.max().item()

        B, N_total, _ = x.shape
        range_tensor = torch.arange(N_total, device=x.device).unsqueeze(0).expand(B, N_total)
        keep_mask = range_tensor < total_seqlens.unsqueeze(1)

        x_packed = x[keep_mask]

        for blk in self.decoder_blocks:
            x_packed = blk(x_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            
        x_packed = self.decoder_norm(x_packed)

        x_split = torch.split(x_packed, total_seqlens.tolist(), dim=0)
        x = torch.nn.utils.rnn.pad_sequence(x_split, batch_first=True)
        if x.shape[1] < N_total:
            padding_len = N_total - x.shape[1]
            zeros = torch.zeros(B, padding_len, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, zeros], dim=1)

        x = x[:, 1:, :] 

        return x

class AdaptiveMAE(nn.Module):
    def __init__(self, img_size=(96, 96, 96), patch_size=(4, 4, 4), in_chans=40,
                 embed_dim=768, depth=12, num_heads=12, qkv_bias=True, qk_norm=False,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mask_ratio=0.75, drop_path_rate=0.1,
                 num_scales=2, method='std', mean=[0.0], std=[1],
                 thresholds=[0.23, 0.23], proj_bias=True,
                 gate_attention='none', patch_norm=True,
                 mixed_patch_embed=None):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        self.patch_norm = patch_norm

        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            qk_norm=qk_norm,
            mixed_patch_embed=mixed_patch_embed,
            gate_attention=gate_attention,
            num_scales=num_scales,
            downstream=False
        )
        self.patch_tokenizer = PatchTokenizer3D(
            num_scales=num_scales,
            base_patch_size=patch_size[0],
            method=method,
            mean=mean,
            std=std,
            thresholds=thresholds,
            image_size=img_size
        )

        self.decoder = MAEDecoder(
            patch_size=patch_size,
            img_size=img_size,
            num_classes=in_chans,
            embed_dim=embed_dim,
            gate_attention=gate_attention,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            proj_bias=proj_bias,
            drop_path_rate=drop_path_rate,
            qk_norm=qk_norm,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        # self.encoder_scale_embed = nn.Parameter(torch.zeros(1, num_scales, embed_dim))
        # torch.nn.init.normal_(self.encoder_scale_embed, std=.02)

    def random_masking(self, x, mask_ratio, input_dict):
        """
        x: (B, N_max, D)
        input_dict: 包含 'seqlens' (List or Tensor)
        """
        B, N, D = x.shape
        device = x.device
        
        seqlens = torch.as_tensor(input_dict['seqlens'], device=device) # (B,)
        
        noise = torch.rand(B, N, device=device) # (B, N)
        
        # Mask Out Padding
        range_tensor = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        padding_mask = range_tensor >= seqlens.unsqueeze(1)
        noise[padding_mask] = 1e9 
        
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascending: small noise (keep) -> large noise (mask) -> inf (padding)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        len_keep_per_batch = (seqlens * (1 - mask_ratio)).long()
        len_keep_per_batch = torch.max(len_keep_per_batch, torch.ones_like(len_keep_per_batch))
        
        # (B, N)
        mask_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        keep_mask = mask_indices < len_keep_per_batch.unsqueeze(1)
        
        # x_shuffled: (B, N, D)
        x_shuffled = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        
        # 0 is keep, 1 is remove. 
        mask_binary = torch.ones([B, N], device=device)
        mask_binary[keep_mask] = 0
        
        mask_binary = torch.gather(mask_binary, dim=1, index=ids_restore)
        
        x_masked_packed = x_shuffled[keep_mask] # (Total_Visible, D)
        
        return x_masked_packed, mask_binary, ids_restore, len_keep_per_batch, ids_shuffle

    def forward_encoder(self, x, input_dict):

        current_img_size = x.shape[2:]

        # Now mixed_patch returns (B, N_max, D) directly
        tokens_batched, _, _, _, _ = self.encoder.mixed_patch(
            x, 
            self.encoder.pos_embed, 
            input_dict, 
            current_img_size=current_img_size
        )

        # Separate CLS and Patches
        cls_token = tokens_batched[:, 0:1, :]
        x_patches = tokens_batched[:, 1:, :]
        
        patch_seqlens = torch.as_tensor(input_dict['seqlens'], device=x.device) - 1
        temp_dict = {'seqlens': patch_seqlens}

        # Random Masking
        x_packed_patches, mask, ids_restore, len_keep_per_batch, ids_shuffle = self.random_masking(x_patches, self.mask_ratio, temp_dict)
        cls_tokens_flat = cls_token.squeeze(1)

        patch_splits = torch.split(x_packed_patches, len_keep_per_batch.tolist(), dim=0)
        
        final_packed_list = []
        for i, split in enumerate(patch_splits):
            final_packed_list.append(cls_tokens_flat[i:i+1])
            final_packed_list.append(split)
            
        x_vis_packed = torch.cat(final_packed_list, dim=0) # (Total_Visible_With_CLS, D)
        
        new_seqlens = len_keep_per_batch + 1
        cu_seqlens = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int32), 
                                new_seqlens.cumsum(0, dtype=torch.int32)])
        max_seqlen = new_seqlens.max().item()
        
        latent_packed_list = self.encoder.forward_features(x_vis_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        
        latent_packed = latent_packed_list[-1]

        max_keep_len = len_keep_per_batch.max().item()
        ids_keep = ids_shuffle[:, :max_keep_len]

        # Unpack Logic:
        latent_splits = torch.split(latent_packed, new_seqlens.tolist(), dim=0)

        latent_padded = torch.nn.utils.rnn.pad_sequence(latent_splits, batch_first=True)
        
        return latent_padded, mask, ids_restore, len_keep_per_batch, ids_keep

    def unfold_3d_to_patches(self, x, patch_size):

        B, C, D, H, W = x.shape
        p_d, p_h, p_w = patch_size
        
        assert D % p_d == 0 and H % p_h == 0 and W % p_w == 0

        x = x.view(B, C, D // p_d, p_d, H // p_h, p_h, W // p_w, p_w)
        
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        
        # Flatten: (B, N_patches, C*pd*ph*pw)
        x = x.view(B, -1, C * p_d * p_h * p_w)
        return x

    def get_ground_truth_patches(self, imgs, patch_size):

        B, C, D, H, W = imgs.shape
        pd, ph, pw = patch_size
        
        # Input: (B, C, D, H, W) -> Unfold -> (B, C, N_patches, pd, ph, pw)
        
        D_grid = D // pd
        H_grid = H // ph
        W_grid = W // pw
        
        # (B, C, D//pd, pd, H//ph, ph, W//pw, pw)
        x = imgs.view(B, C, D_grid, pd, H_grid, ph, W_grid, pw)
        # (B, D_grid, H_grid, W_grid, C, pd, ph, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        # (B, N_total, Flatten_Pixels)
        x = x.view(B, -1, C * pd * ph * pw)
        return x

    def forward_loss(self, imgs, hidden_states, mask, input_dict, corr_weight=1):
        """
        Modified Loss Calculation:
        Strictly matches the scale of the predictor to the scale of the patch.
        """
        patch_coords = input_dict['patch_coords']       # (B, N_max, 3)
        patch_scales = input_dict['patch_scale_indices'] # (B, N_max)
        
        B, N_max, D_emb = hidden_states.shape
        sim_loss = torch.tensor(0.0, device=imgs.device)
        total_loss = torch.tensor(0.0, device=imgs.device)
        total_masked_tokens = 0
        
        target_dict = {}
        
        # Scale 0 Targets (Base Size)
        target_dict[0] = self.get_ground_truth_patches(imgs, self.patch_size)
        
        # Scale 1 Targets (2x Base Size)
        if self.decoder.predictors and len(self.decoder.predictors) > 1:
            large_patch_size = tuple([p * 2 for p in self.patch_size])
            target_dict[1] = self.get_ground_truth_patches(imgs, large_patch_size)

        is_padding = (patch_scales == -1)
        loss_mask = (mask == 1) & (~is_padding)

        num_scales = len(self.decoder.predictors)
        
        for s_idx in range(num_scales):

            current_scale_mask = loss_mask & (patch_scales == s_idx)
            
            if not current_scale_mask.any():
                continue
                
            # subset_preds: (M, Embed_Dim) -> Project -> (M, Pixels)
            subset_hidden = hidden_states[current_scale_mask]
            subset_preds = self.decoder.predictors[s_idx](subset_hidden)
            
            subset_coords = patch_coords[current_scale_mask] # (M, 3) (z, y, x)
            subset_batch_idx = torch.arange(B, device=imgs.device).unsqueeze(1).expand(B, N_max)[current_scale_mask]
            
            current_p_size = [p * (2**s_idx) for p in self.patch_size]
            D_grid = imgs.shape[2] // current_p_size[0]
            H_grid = imgs.shape[3] // current_p_size[1]
            W_grid = imgs.shape[4] // current_p_size[2]
            
            gz = subset_coords[:, 0] // current_p_size[0]
            gy = subset_coords[:, 1] // current_p_size[1]
            gx = subset_coords[:, 2] // current_p_size[2]
            
            linear_idx = gz * (H_grid * W_grid) + gy * W_grid + gx
            
            subset_targets = target_dict[s_idx][subset_batch_idx, linear_idx]
            
            if self.patch_norm:
                mean = subset_targets.mean(dim=-1, keepdim=True)
                var = subset_targets.var(dim=-1, keepdim=True)
                subset_targets = (subset_targets - mean) / (torch.sqrt(var) + 1.e-5)
            
            mse = (subset_preds - subset_targets) ** 2
            sim_loss += mse.mean(dim=-1).sum() 
            
            total_masked_tokens += subset_targets.shape[0]

        total_loss = sim_loss / total_masked_tokens
        return total_loss

    def forward(self, imgs):
    
        input_dict = self.patch_tokenizer(imgs)

        latent, mask, ids_restore, len_keep, _ = self.forward_encoder(imgs, input_dict)

        hidden_states = self.decoder(latent, ids_restore, input_dict, len_keep=len_keep)
        
        loss = self.forward_loss(imgs, hidden_states, mask, input_dict)

        return loss, mask, hidden_states
    

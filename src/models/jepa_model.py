import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer
from .vit_components import Block
from .patch_embed_3d import TokenizedZeroConvPatchAttn3D
from .lejepa.lejepa import univariate, multivariate


class JEPAPredictor(nn.Module):

    def __init__(self, patch_size, img_size, embed_dim=768, predictor_embed_dim=384,
                 predictor_depth=6, predictor_num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 gate_attention='elementwise', qk_norm=True, drop_path_rate=0.1,proj_bias=True,
                 num_scales=2):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.predictor_embed_dim = predictor_embed_dim

        self.encoder_to_predictor = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )

        self.pos_embed_vol = nn.Parameter(
            torch.zeros(1, predictor_embed_dim, *self.grid_size)
        )
        torch.nn.init.trunc_normal_(self.pos_embed_vol, std=.02)

        self.scale_embed = nn.Parameter(torch.zeros(1, num_scales, predictor_embed_dim))
        torch.nn.init.normal_(self.scale_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, predictor_depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=predictor_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=qk_norm,
                drop_path=dpr[i],
                proj_bias=proj_bias,
                gate_attention=gate_attention,
                norm_layer=norm_layer
            )
            for i in range(predictor_depth)])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        
        self.predictor_pred = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

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

    def get_pos_and_scale_embed(self, patch_coords, patch_scale_indices, current_img_size):

        B, N_max, _ = patch_coords.shape
        D_img, H_img, W_img = current_img_size 
        device = patch_coords.device

        safe_indices = patch_scale_indices.clamp(min=0)
        scale_factors = (2 ** safe_indices).unsqueeze(-1)
        patch_sizes = torch.tensor(self.patch_size, device=device).view(1, 1, 3) * scale_factors
        centers = patch_coords + patch_sizes / 2.0
        
        img_size_tensor = torch.tensor([D_img, H_img, W_img], device=device).view(1, 1, 3)
        norm_coords = 2.0 * (centers / img_size_tensor) - 1.0
        
        grid = norm_coords.flip(-1).view(B, 1, 1, N_max, 3)
        
        pos_vol = self.pos_embed_vol.expand(B, -1, -1, -1, -1)
        sampled_pos = F.grid_sample(
            pos_vol, grid, 
            mode='bilinear', padding_mode='border', align_corners=False
        )
        sampled_pos = sampled_pos.squeeze(2).squeeze(2).transpose(1, 2) # (B, N, Dim)
        
        valid_indices = patch_scale_indices.clamp(min=0)
        scale_embeds = self.scale_embed[0, valid_indices, :] 
        
        return sampled_pos, scale_embeds

    def forward(self, x, ids_restore, input_dict, len_keep=None, current_img_size=None):

        x = self.encoder_to_predictor(x)

        cls_token = x[:, :1, :]
        x_patches = x[:, 1:, :] # (B, N_vis_max, D)
        
        B, N_vis_max, D = x_patches.shape
        N_max = ids_restore.shape[1]
        
        x_full_shuffled = self.mask_token.to(dtype=x.dtype).repeat(B, N_max, 1)
        
        if len_keep is not None:
            range_tensor = torch.arange(N_vis_max, device=x.device).unsqueeze(0).expand(B, -1)
            valid_mask = range_tensor < len_keep.unsqueeze(1)
            x_full_shuffled[:, :N_vis_max, :][valid_mask] = x_patches[valid_mask]
        else:
            x_full_shuffled = torch.cat([x_patches, self.mask_token.repeat(B, N_max - N_vis_max, 1)], dim=1)
        
        x_full = torch.gather(x_full_shuffled, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        patch_coords = input_dict['patch_coords']       
        patch_scales = input_dict['patch_scale_indices'] 
        
        if current_img_size is None:
             current_img_size = self.img_size 

        pos_embed, scale_embed = self.get_pos_and_scale_embed(patch_coords, patch_scales, current_img_size)
        
        x_full = x_full + pos_embed + scale_embed
        
        x = torch.cat([cls_token, x_full], dim=1)

        total_seqlens = torch.as_tensor(input_dict['seqlens'], device=x.device)
        cu_seqlens = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int32), 
                                total_seqlens.cumsum(0, dtype=torch.int32)])
        max_seqlen = total_seqlens.max().item()

        B, N_total, _ = x.shape
        range_tensor = torch.arange(N_total, device=x.device).unsqueeze(0).expand(B, N_total)
        keep_mask = range_tensor < total_seqlens.unsqueeze(1)

        x_packed = x[keep_mask]

        for blk in self.predictor_blocks:
            x_packed = blk(x_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            
        x_packed = self.predictor_norm(x_packed)
        x_packed = self.predictor_pred(x_packed)

        x_split = torch.split(x_packed, total_seqlens.tolist(), dim=0)
        x = torch.nn.utils.rnn.pad_sequence(x_split, batch_first=True)
        if x.shape[1] < N_total:
            padding_len = N_total - x.shape[1]
            zeros = torch.zeros(B, padding_len, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, zeros], dim=1)

        x = x[:, 1:, :] 
        
        return x


class AdaptiveJEPA(nn.Module):
    def __init__(self, img_size=(96, 96, 96), patch_size=(4, 4, 4), in_chans=40,
                 embed_dim=768, depth=12, num_heads=12, qkv_bias=True, qk_norm=True,
                 predictor_embed_dim=384, predictor_depth=6, predictor_num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mask_ratio=0.75,
                 num_scales=2, method='std', drop_path_rate=0.1, proj_bias=True, 
                 thresholds=[0.23], n_points=17, num_slices=2048, lamda=0.05, alpha=0.5,
                 use_flatten_tokens=True, proj_dim=128,
                 gate_attention='none', global_pool='', use_patch_loss=True, sample_ratio=0.2,
                 mixed_patch_embed=None):
        super().__init__()

        self.mask_ratio = mask_ratio
        
        # 1. Shared Encoder (Student & Teacher share this)
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            mixed_patch_embed=mixed_patch_embed,
            gate_attention=gate_attention,
            global_pool=global_pool,
            drop_path_rate=drop_path_rate,
            method=method,
            proj_bias=proj_bias,
            thresholds=thresholds,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            downstream=False 
        )

        # 2. Light Predictor
        self.predictor = JEPAPredictor(
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            predictor_depth=predictor_depth,
            predictor_num_heads=predictor_num_heads,
            gate_attention=gate_attention,
            proj_bias=proj_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
            num_scales=num_scales
        )

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048), 
            nn.BatchNorm1d(2048),       
            nn.Linear(2048, proj_dim)   
        )
        # Choose a univariate test 
        univariate_test = univariate.EppsPulley(n_points=n_points)

        # Create the multivariate slicing test
        self.sigreg_loss = multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices
        )

        self.lamda = lamda
        self.alpha = alpha
        self.use_patch_loss = use_patch_loss
        self.sample_ratio = sample_ratio

        self.use_flatten_tokens = use_flatten_tokens

    def _sample_valid_patches(self, x, lengths, sample_ratio):

        B, N, D = x.shape
        device = x.device
        
        range_tensor = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device).unsqueeze(1)
        elif lengths.ndim == 1:
            lengths = lengths.unsqueeze(1)
            
        valid_mask = (range_tensor < lengths) & (range_tensor > 0)
        
        valid_tokens = x[valid_mask]
        
        total_valid = valid_tokens.shape[0]
        
        if total_valid == 0:
            return valid_tokens
        
        if sample_ratio >= 1.0:
            return valid_tokens
            
        num_to_sample = int(total_valid * sample_ratio)
        
        if num_to_sample == 0:
            num_to_sample = 1
            
        perm = torch.randperm(total_valid, device=device)[:num_to_sample]
        
        return valid_tokens[perm]

    def forward_encoder_context(self, x, input_dict):

        current_img_size = x.shape[2:]

        # 1. Patch Embedding
        tokens_batched, _, _, _, _ = self.encoder.mixed_patch(
            x, 
            self.encoder.pos_embed, 
            input_dict, 
            current_img_size=current_img_size
        )
        
        cls_token = tokens_batched[:, 0:1, :]
        x_patches = tokens_batched[:, 1:, :]

        patch_seqlens = torch.as_tensor(input_dict['seqlens'], device=x.device) - 1
        temp_dict = {'seqlens': patch_seqlens}
        
        # 2. Random Masking
        x_packed_patches, mask, ids_restore, len_keep_per_batch = self.random_masking(x_patches, self.mask_ratio, temp_dict)
        
        # 3. Pack Visible Tokens + CLS
        cls_tokens_flat = cls_token.squeeze(1)
        patch_splits = torch.split(x_packed_patches, len_keep_per_batch.tolist(), dim=0)
        
        final_packed_list = []
        for i, split in enumerate(patch_splits):
            final_packed_list.append(cls_tokens_flat[i:i+1])
            final_packed_list.append(split)
        
        x_vis_packed = torch.cat(final_packed_list, dim=0) 
        
        # 4. Encoder Forward (Flash Attn Varlen)
        new_seqlens = len_keep_per_batch + 1
        cu_seqlens = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int32), 
                                new_seqlens.cumsum(0, dtype=torch.int32)])
        max_seqlen = new_seqlens.max().item()
        
        latent_packed = self.encoder.forward_features(x_vis_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        
        # 5. Unpack to (B, N_visible_max, D)
        latent_splits = torch.split(latent_packed, new_seqlens.tolist(), dim=0)
        latent_padded = torch.nn.utils.rnn.pad_sequence(latent_splits, batch_first=True)
        
        return latent_padded, mask, ids_restore, len_keep_per_batch

    def random_masking(self, x, mask_ratio, input_dict):
        B, N, D = x.shape
        device = x.device
        seqlens = torch.as_tensor(input_dict['seqlens'], device=device) # Patch count
        
        noise = torch.rand(B, N, device=device)
        range_tensor = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        padding_mask = range_tensor >= seqlens.unsqueeze(1)
        noise[padding_mask] = 1e9 
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        len_keep_per_batch = (seqlens * (1 - mask_ratio)).long()
        len_keep_per_batch = torch.max(len_keep_per_batch, torch.ones_like(len_keep_per_batch))
        
        mask_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        keep_mask = mask_indices < len_keep_per_batch.unsqueeze(1)
        
        x_shuffled = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        
        mask_binary = torch.ones([B, N], device=device)
        mask_binary[keep_mask] = 0
        mask_binary = torch.gather(mask_binary, dim=1, index=ids_restore)
        
        x_masked_packed = x_shuffled[keep_mask]
        
        return x_masked_packed, mask_binary, ids_restore, len_keep_per_batch

    def forward_loss(self, target_feats, pred_feats, mask, input_dict):

        target_patches = target_feats[:, 1:, :] 
        
        B, N_tgt, D = target_patches.shape
        B_pred, N_pred, D_pred = pred_feats.shape
        
        assert N_tgt == N_pred, \
            f"Shape mismatch! Target patches: {N_tgt}, Pred patches: {N_pred}. Check encoder padding or predictor logic."
        
        patch_scales = input_dict['patch_scale_indices'] # (B, N_max)

        is_padding = (patch_scales == -1)
        
        loss = (pred_feats - target_patches) ** 2
        loss = loss.mean(dim=-1) # (B, N)
        valid_loss_mask = (mask.bool()) & (~is_padding)

        sum_mask = valid_loss_mask.sum()
        if sum_mask > 0:
            loss = (loss * valid_loss_mask).sum() / sum_mask
        else:
            loss = loss.sum() * 0.0 

        return loss

    def forward(self, imgs):

        current_img_size = imgs.shape[2:]

        with torch.no_grad():
            target_full, input_dict = self.encoder(imgs)
        
        target_full = F.layer_norm(target_full, target_full.shape[-1:])

        context_latent, mask, ids_restore, len_keep = self.forward_encoder_context(imgs, input_dict)
        context_cls = context_latent[:, 0, :] # Shape: (B, D)
        context_lengths = len_keep + 1


        pred_feats = self.predictor(context_latent, ids_restore, input_dict, len_keep=len_keep, current_img_size=current_img_size)

        if self.use_patch_loss:
            context_patches_flat = self._sample_valid_patches(
                context_latent, 
                context_lengths, 
                self.sample_ratio 
            )
            sigreg_input = context_patches_flat

        elif self.use_flatten_tokens:
            B, N_max, C = context_latent.shape
            range_tensor = torch.arange(N_max, device=context_latent.device).unsqueeze(0)
            valid_mask = range_tensor < context_lengths.unsqueeze(1)
            sigreg_input = context_latent[valid_mask]
        else:
            sigreg_input = context_cls

        sigreg_input = self.projector(sigreg_input)

        if self.use_flatten_tokens:
            sigreg_loss = (self.sigreg_loss(sigreg_input).mean()) / B
        else:
            sigreg_loss = self.sigreg_loss(sigreg_input)
        
        sim_loss = self.forward_loss(target_full, pred_feats, mask, input_dict)

        loss = (1 - self.lamda) * sim_loss + self.lamda * sigreg_loss

        aliment_metric = loss / (self.lamda ** self.alpha)

        return loss, sim_loss, sigreg_loss, aliment_metric


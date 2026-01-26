""" Vision Transformer (ViT) in PyTorch

- based on the timm implementation, removing unused components WIP
"""
import math
from functools import partial
from typing import  Callable, Optional, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.layers import Mlp, \
    trunc_normal_, \
    get_act_layer, get_norm_layer, LayerType
from timm.models.vision_transformer import get_init_weights_vit, named_apply
from .vit_components import  Block
from .patch_embed_3d import PatchEmbed3D 
from .patch_tokenizer_3d import PatchTokenizer3D


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int, int]] = 96,
            patch_size: Union[int, Tuple[int, int, int]] = 4,
            in_chans: int = 40,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = True,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            mixed_patch_embed: Optional[Callable] = None,
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = nn.LayerNorm,
            act_layer: Optional[LayerType] = nn.GELU,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            num_scales: Optional[int] = 2,
            thresholds: List[float] = [0.23],
            alpha_schedule: Optional[bool] = False,
            downstream: Optional[bool] = True,
            post_train: bool = False,
            fusion_mode: str = 'none',
            enable_llm: bool = False,
            gate_attention: str = 'none',
            method: str = 'std',
            freeze_backbone: bool = False
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.img_size = img_size
        self.patch_size = patch_size

        self.num_classes = num_classes
        self.patch_sizes = [patch_size * 2**i for i in range(num_scales)]
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.downstream = downstream
        self.enable_llm = enable_llm
        self.post_train = post_train

        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        # Basic patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  
            **embed_args,
        )
        
        self.num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = self.num_patches if no_embed_class else self.num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.patch_tokenizer = PatchTokenizer3D(
            num_scales=num_scales,
            base_patch_size=patch_size,
            method=method,
            thresholds=thresholds,
            image_size=img_size
        )

        # Mixed patch embedding if specified
        if mixed_patch_embed is not None:
            self.mixed_patch = mixed_patch_embed(
                image_size=img_size, 
                patch_size=patch_size, 
                embed_dim=embed_dim,
                num_scales=num_scales,
                thresholds=thresholds,
                in_chans=in_chans,
            )
            self.mixed_patch.no_embed_class = self.no_embed_class
            
            if alpha_schedule:
                print('Using mask scheduling !!!')
        else:
            self.mixed_patch = None

        self.depth = depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                gate_attention=gate_attention
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()
        self.classifier = nn.Linear(1024,2)
        if self.downstream:
            self.fusion_mode = fusion_mode
            if self.fusion_mode == '1':
                self.layer_weights = nn.Parameter(torch.ones(8)) 
                self.fusion_dim = embed_dim * 5 
            elif self.fusion_mode == '2':
                self.fusion_dim = embed_dim * 4
            elif self.fusion_mode == '3':
                self.layer_weights = nn.Parameter(torch.ones(4))
                self.fusion_dim = embed_dim
            if self.fusion_mode != 'none' and self.fusion_mode != 'pool_mean' and self.fusion_dim != self.embed_dim:
                self.bottennet = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
            self.fc_norm = norm_layer(self.embed_dim) if final_norm and use_fc_norm else nn.Identity()
            self.head_drop = nn.Dropout(drop_rate)

            if not freeze_backbone:
                self.head = nn.Sequential(
                    nn.Linear(self.embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, num_classes)
                )
            else:
                self.head = nn.Linear(self.embed_dim, num_classes)


        if self.post_train:
            self.fmri_proj = nn.Sequential(
                             nn.Linear(embed_dim, 384),
                             nn.LayerNorm(384),
                             nn.GELU(),
                             nn.Linear(384, 256)
            )
            if self.downstream:
                self.head_classify = nn.Linear(1024, 2)

        self.init_multiscale_patch_embed()
        if weight_init != 'skip':
            self.init_weights(weight_init)
        if self.mixed_patch is not None and hasattr(self.mixed_patch, 'zero_conv'):
            print("Re-initializing zero_conv to zeros (fixing overwrite)...")
            nn.init.zeros_(self.mixed_patch.zero_conv.weight)
            nn.init.zeros_(self.mixed_patch.zero_conv.bias)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def init_multiscale_patch_embed(self):
        # Use FALSE for antialiasing, following timm    
        # (https://github.com/huggingface/pytorch-image-models/blob/d81da93c1640a504977b0ee494791e5c634ec63c/timm/models/vision_transformer.py#L2259)
       
        self.mixed_patch.patch_embed = self.patch_embed
        self.mixed_patch.cls_token = self.cls_token
        # self.cls_token = None # remove this.
        #self.mixed_patch.pos_drop = self.pos_drop

        # Store only the base position embedding
        self.mixed_patch.base_pos_embed = self.pos_embed
        
    def forward_features(self, x: torch.Tensor, cu_seqlens=None, max_seqlen=None) -> torch.Tensor:
        x = self.norm_pre(x)
        all_layer_outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            if i == len(self.blocks) - 1:
                curr_out = self.norm(x)
            else:
                curr_out = x
            all_layer_outputs.append(curr_out)
        return all_layer_outputs


    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode != 'none' and self.fusion_mode != 'pool_mean' and self.fusion_dim != self.embed_dim:
            x = self.bottennet(x)
        x = self.fc_norm(x)
        logits = self.head(x)
        return logits

    def _extract_pooled_features(self, x_out_list: List[torch.Tensor], input_dict, cu_seqlens, seqlens):
        """ Helper to pool features from all layers into (B, D) tensors """
        pooled_layers = []
        for x_out in x_out_list:
            # Apply pooling logic to each layer's output
            if self.global_pool == 'token':
                # Assuming packed sequence with CLS at the start of each seq
                cls_indices = cu_seqlens[:-1].long()
                pooled = x_out[cls_indices] # (B, D)
            
            elif self.global_pool == '':
                # Return full sequence (not supported for this fusion type easily)
                raise NotImplementedError("Global pool '' (None) not supported with layer fusion")
            
            else:
                # avg, max, avgmax
                lengths = seqlens.tolist()
                splits = torch.split(x_out, lengths, dim=0)
                
                batch_pooled = []
                for seq in splits:
                    if seq.shape[0] > 1:
                        patch_tokens = seq[1:] 
                        if self.global_pool == 'avg':
                            batch_pooled.append(patch_tokens.mean(dim=0))
                        elif self.global_pool == 'max':
                            batch_pooled.append(patch_tokens.amax(dim=0))
                        elif self.global_pool == 'avgmax':
                            batch_pooled.append(0.5 * (patch_tokens.mean(dim=0) + patch_tokens.amax(dim=0)))
                        else:
                            batch_pooled.append(patch_tokens.mean(dim=0))
                    else:
                        # Fallback if only CLS exists
                         batch_pooled.append(seq[0])
                
                pooled = torch.stack(batch_pooled) # (B, D)
            
            pooled_layers.append(pooled)
            
        return pooled_layers

    def forward(self, x) -> torch.Tensor:

        input_dict = self.patch_tokenizer(x)

        target_dtype = self.pos_embed.dtype
        with torch.cuda.amp.autocast(enabled=True, dtype=target_dtype):
            if x.dtype != target_dtype:
                x = x.to(dtype=target_dtype)
            
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    if v.dtype != target_dtype:
                        input_dict[k] = v.to(dtype=target_dtype)
        
        current_img_size = x.shape[2:]
        x, cu_seqlens, max_seqlen, _, _ = self.mixed_patch(
            x, 
            self.pos_embed, 
            input_dict, 
            current_img_size=current_img_size
        )
        x = x.to(dtype=target_dtype)

        B, N_max, D = x.shape
        seqlens = torch.as_tensor(input_dict['seqlens'], device=x.device) # (B,)

        cu_seqlens = torch.cat([
            torch.zeros(1, device=x.device, dtype=torch.int32),
            seqlens.cumsum(0, dtype=torch.int32)
        ]).contiguous()

        max_seqlen = seqlens.max().item()

        mask_indices = torch.arange(N_max, device=x.device).unsqueeze(0).expand(B, N_max)
        valid_mask = mask_indices < seqlens.unsqueeze(1)

        x_packed = x[valid_mask] 

        # Run the model.
        with torch.cuda.amp.autocast(dtype=target_dtype):
            all_layer_feats = self.forward_features(x_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        if self.enable_llm:
            return all_layer_feats, cu_seqlens, max_seqlen
        
        pooled_feats = self._extract_pooled_features(all_layer_feats, input_dict, cu_seqlens, seqlens)

        if self.post_train:
            x = pooled_feats[-1]
            fmri_feat = self.fmri_proj(x)
            if self.downstream:
                fmri_feat = self.head_classify(fmri_feat)
            return fmri_feat

        if self.fusion_mode == '1':
            # 1. Layers 1-8 (indices 0-7): Weighted Sum
            norm_weights = F.softmax(self.layer_weights, dim=0)
            low_level_feat = 0
            for i in range(8):
                low_level_feat += norm_weights[i] * pooled_feats[i]
            high_level_feats = pooled_feats[8:] # [layer8, layer9, layer10, layer11]
            high_level_feat = torch.cat(high_level_feats, dim=-1) # (B, 4*D)
            final_feat = torch.cat([low_level_feat, high_level_feat], dim=-1) # (B, 5*D)
            logit = self.forward_head(final_feat)
        elif self.fusion_mode == '2':
            high_level_feats = pooled_feats[8:] 
            high_level_feat = torch.cat(high_level_feats, dim=-1) # (B, 4*D)
            logit = self.forward_head(high_level_feat)
        elif self.fusion_mode == '3':
            norm_weights = F.softmax(self.layer_weights, dim=0)
            feat = 0
            last4_feats = pooled_feats[-4:]
            for i in range(4):
                feat += norm_weights[i] * last4_feats[i]
            logit = self.forward_head(feat)
        elif self.fusion_mode == 'none':
            x = pooled_feats[-1]
            logit = self.forward_head(x)
        elif self.fusion_mode == 'pool_mean':
            x = pooled_feats[-1]
            logit = self.forward_head(x)

        return logit
    
    
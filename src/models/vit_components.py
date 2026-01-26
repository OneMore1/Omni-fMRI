from typing import Optional, Type, Final
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import Mlp, DropPath
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            gate_attention: str = 'none'
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gate_attention = gate_attention

        self.q_aug_dim = 0
        if gate_attention == 'headwise':
            self.q_aug_dim = self.num_heads
        elif gate_attention == 'elementwise':
            self.q_aug_dim = self.dim
        total_dim = dim * 3 + self.q_aug_dim

        self.qkv = nn.Linear(dim, total_dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if torch.isnan(x).any():
             print("!!! [Attention Input] x contains NaN!")

        is_packed = x.ndim == 2
        if is_packed:
            Total, C = x.shape
            B = 1 # Dummy
            N = Total
        else:
            B, N, C = x.shape

        qkv = self.qkv(x)

        if torch.isnan(qkv).any():
             print("!!! [QKV Output] contains NaN! (Weights might be corrupted)")
        
        # qkv shape: (B, N, Total_Dim) or (Total, Total_Dim)
        
        q_aug, k, v = torch.split(qkv, [self.dim + self.q_aug_dim, self.dim, self.dim], dim=-1)

        if self.gate_attention == 'headwise':
            q, gate = torch.split(q_aug, [self.dim, self.num_heads], dim=-1)
            # Gate: (..., num_heads) -> (..., num_heads, 1)
            gate = gate.unsqueeze(-1) 
        elif self.gate_attention == 'elementwise':
            q, gate = torch.split(q_aug, [self.dim, self.dim], dim=-1)
            # Gate: (..., dim) -> (..., num_heads, head_dim)
            gate = gate.reshape(*gate.shape[:-1], self.num_heads, self.head_dim)
        else:
            q = q_aug
            gate = None

        # --- Flash Attention Path (Varlen) ---
        if HAS_FLASH_ATTN and cu_seqlens is not None and max_seqlen is not None:

            cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
            if torch.isnan(q).any() or torch.isinf(q).any():
                print("!!! Q contains NaN/Inf BEFORE FlashAttn")

            # Reshape Q, K, V to (Total, Num_Heads, Head_Dim)
            q = q.reshape(-1, self.num_heads, self.head_dim)
            k = k.reshape(-1, self.num_heads, self.head_dim)
            v = v.reshape(-1, self.num_heads, self.head_dim)

            if torch.isnan(q).any(): print("!!! Q is NaN BEFORE Norm")

            if gate is not None:
                if self.gate_attention == 'headwise':
                    gate = gate.reshape(-1, self.num_heads, 1)
                else:
                    gate = gate.reshape(-1, self.num_heads, self.head_dim)

            q, k = self.q_norm(q), self.k_norm(k)

            if torch.isnan(q).any(): 
                print(f"!!! Q contains NaN AFTER Norm. Q_Norm Weight: {self.q_norm.weight.mean()}")

            # Cast to bf16/fp16 for flash attention
            target_dtype = q.dtype
            if q.dtype == torch.float32:
                target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                q = q.to(target_dtype)
                k = k.to(target_dtype)
                v = v.to(target_dtype)

            cu_seqlens = cu_seqlens.to(torch.int32)
            
            # Flash Attention Call
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            )
            
            if out.dtype != x.dtype:
                out = out.to(x.dtype)

            if gate is not None:
                out = out * F.sigmoid(gate)
            
            # Reshape back to input shape
            if is_packed:
                out = out.reshape(Total, C)
            else:
                out = out.reshape(B, N, C)

        # --- Standard Attention Path ---
        else:
            if is_packed:
                raise ValueError("Standard Attention does not support packed sequences directly. Use (B, N, C) input or enable Flash Attention.")

            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = torch.zeros_like(key_padding_mask, dtype=q.dtype)
                attn_mask.masked_fill_(key_padding_mask, float("-inf"))
                # Reshape for broadcasting: (B, Num_Heads, Q_len, K_len)
                attn_mask = attn_mask.view(B, 1, 1, N)

            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            q, k = self.q_norm(q), self.k_norm(k)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

            out = out.transpose(1, 2) # (B, N, H, D)
            
            if gate is not None:
                # Gate: (B, N, H, 1) or (B, N, H, D) -> Permute -> (B, H, N, D/1)
                gate = gate.permute(0, 2, 1, 3)
                out = out * F.sigmoid(gate)
                
            out = out.reshape(B, N, C)

        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            gate_attention: str = 'none'
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            gate_attention=gate_attention
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None, key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, key_padding_mask=key_padding_mask )))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


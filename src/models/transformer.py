from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

from flash_attn import flash_attn_varlen_func


class Mlp(nn.Module):
    """FFN for transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # TODO: update this to the more standard TIMM
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """MHA implementation for VIT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = 1024,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        #head_dim = 85
        all_head_dim = head_dim * self.num_heads
        #ipdb.set_trace()
        #all_head_dim = 1020
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        #self.qkv = nn.Linear(dim, 1020, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash_attn = use_flash_attn

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None, cu_seqlens: Optional[Tensor] = None, max_seqlen: Optional[int] = None) -> Tensor:
        """Scaled DP attention with flash-attn varlen support."""
        B, N, C = x.shape
        qkv_bias = None
        # AVION uses a QKV bias. We'll stick with this.
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, 
                 torch.zeros_like(self.v_bias, requires_grad=False), 
                 self.v_bias)
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        if self.use_flash_attn and cu_seqlens is not None and max_seqlen is not None:
            # Flash-attn varlen API: requires flattened inputs
            # q, k, v: (B, num_heads, N, head_dim) -> (total_tokens, num_heads, head_dim)
            q = q.transpose(1, 2).reshape(-1, self.num_heads, q.shape[-1])
            k = k.transpose(1, 2).reshape(-1, self.num_heads, k.shape[-1])
            v = v.transpose(1, 2).reshape(-1, self.num_heads, v.shape[-1])
            
            x = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attn_drop_rate if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            )
            # x: (total_tokens, num_heads, head_dim) -> (B, N, num_heads * head_dim)
            x = x.reshape(B, N, -1)
        elif self.use_flash_attn:
            # Standard flash attention (non-varlen)
            from flash_attn import flash_attn_func
            q = q.transpose(1, 2)  # (B, N, num_heads, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop_rate if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            )
            x = x.reshape(B, N, -1)
        else:
            # Non-optimized version.
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2)
            x = x.reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        use_flash_attn=True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_head_dim=attn_head_dim,
            use_flash_attn=use_flash_attn
        )
        
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity() # DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_rate
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None, cu_seqlens=None, max_seqlen=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_mask, cu_seqlens, max_seqlen))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), attn_mask, cu_seqlens, max_seqlen))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    

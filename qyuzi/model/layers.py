import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import config

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.cos_cached.shape[0]:
             self._set_cos_sin_cache(seq_len)
             
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Context32KScaling(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int = 32768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        slopes = torch.Tensor(self._get_alibi_slopes(num_heads))
        self.register_buffer('alibi_slopes', slopes)
        
    def _get_alibi_slopes(self, num_heads: int):
        def get_slopes(n):
            return [2 ** (-8 * i / n) for i in range(1, n + 1)]
        return get_slopes(num_heads)
    
    def get_alibi_bias(self, seq_len: int) -> torch.Tensor:
        context_position = torch.arange(seq_len)[:, None].to(self.alibi_slopes.device)
        memory_position = torch.arange(seq_len)[None, :].to(self.alibi_slopes.device)
        relative_position = memory_position - context_position
        relative_position = relative_position.unsqueeze(0).expand(self.num_heads, -1, -1)
        alibi = relative_position * self.alibi_slopes.view(-1, 1, 1)
        return alibi
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = q.shape
        nh = self.num_heads
        hd = self.head_dim
        q = q.view(B, T, nh, hd)
        k = k.view(B, T, nh, hd)
        v = v.view(B, T, nh, hd)
        
        cos, sin = self.rotary(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        alibi_bias = self.get_alibi_bias(T).to(q.device)
        attn_mask = alibi_bias.unsqueeze(0)
        
        if mask is not None:
             if mask.dim() == 2:
                 mask = mask.unsqueeze(0).unsqueeze(0)
             attn_mask = attn_mask + mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else config.DROPOUT_RATE)
        
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return out

class RecurrentGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.candidate = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_prev is None or h_prev.shape[0] != x.shape[0]:
            h_prev = torch.zeros_like(x)
            
        combined = torch.cat([x, h_prev], dim=-1)
        z = torch.sigmoid(self.gate(combined))
        r = torch.sigmoid(self.update(combined))
        h_candidate = torch.tanh(self.candidate(torch.cat([x, r * h_prev], dim=-1)))
        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new

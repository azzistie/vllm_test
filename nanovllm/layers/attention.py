import importlib.util
import sys

import torch
from torch import nn
from torch.nn import functional as F

from nanovllm.utils.context import get_context


USE_FLASH_ATTENTION = (
    sys.platform != "win32"
    and importlib.util.find_spec("triton") is not None
    and importlib.util.find_spec("flash_attn") is not None
)


class TorchAttention(nn.Module):
    """Eager SDPA implementation using the engine's paged KV cache."""

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def cached_kv(self, block_table, length):
        block_size = self.k_cache.size(1)
        blocks = block_table[:(length + block_size - 1) // block_size].long()
        return tuple(cache[blocks].flatten(0, 1)[:length]
                     for cache in (self.k_cache, self.v_cache))

    def attend(self, q, k, v):
        nq, nk = q.size(0), k.size(0)
        # Cached prefixes require a lower-right causal mask. A single decode
        # query can attend to every existing key, including the newest token.
        mask = None
        if 1 < nq < nk:
            mask = torch.arange(nk, device=q.device)[None, :] <= (
                torch.arange(nq, device=q.device)[:, None] + nk - nq
            )
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=mask, is_causal=nq == nk and nq > 1, scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)

    def forward(self, q, k, v):
        context = get_context()
        if self.k_cache.numel():
            slots = context.slot_mapping.long()
            valid = slots >= 0
            for cache, values in ((self.k_cache, k), (self.v_cache, v)):
                cache.view(-1, self.num_kv_heads, self.head_dim)[slots[valid]] = values[valid]
        outputs = []
        if context.is_prefill:
            q_bounds = context.cu_seqlens_q.tolist()
            k_bounds = context.cu_seqlens_k.tolist()
            for i, (start, end) in enumerate(zip(q_bounds, q_bounds[1:])):
                if context.block_tables is None:
                    ki, vi = k[k_bounds[i]:k_bounds[i + 1]], v[k_bounds[i]:k_bounds[i + 1]]
                else:
                    ki, vi = self.cached_kv(context.block_tables[i], k_bounds[i + 1] - k_bounds[i])
                outputs.append(self.attend(q[start:end], ki, vi))
        else:
            for i, length in enumerate(context.context_lens.tolist()):
                ki, vi = self.cached_kv(context.block_tables[i], length)
                outputs.append(self.attend(q[i:i + 1], ki, vi))
        return torch.cat(outputs, dim=0)


if USE_FLASH_ATTENTION:
    from nanovllm.layers.flash_attention import Attention
else:
    Attention = TorchAttention

import unittest

import torch

from nanovllm.layers.attention import TorchAttention
from nanovllm.utils.context import reset_context, set_context


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.attn = TorchAttention(4, 8, 8 ** -0.5, 2)

    def tearDown(self):
        reset_context()

    def reference(self, q, k, v):
        k, v = (x.repeat_interleave(2, dim=1) for x in (k, v))
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.attn.scale
        mask = torch.arange(k.size(0))[None, :] > (
            torch.arange(q.size(0))[:, None] + k.size(0) - q.size(0)
        )
        scores.masked_fill_(mask, float("-inf"))
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)

    def test_variable_length_prefill(self):
        q, k, v = torch.randn(7, 4, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8)
        bounds = torch.tensor([0, 3, 7], dtype=torch.int32)
        set_context(True, cu_seqlens_q=bounds, cu_seqlens_k=bounds)
        expected = torch.cat([self.reference(q[:3], k[:3], v[:3]),
                              self.reference(q[3:], k[3:], v[3:])])
        torch.testing.assert_close(self.attn(q, k, v), expected)

    def test_paged_prefix_and_decode(self):
        q, k, v = torch.randn(6, 4, 8), torch.randn(6, 2, 8), torch.randn(6, 2, 8)
        self.attn.k_cache = torch.zeros(3, 2, 2, 8)
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)
        table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
        slots = torch.tensor([4, 5, 0, 1, 2, 3])
        for cache, values in ((self.attn.k_cache, k), (self.attn.v_cache, v)):
            cache.view(-1, 2, 8)[slots[:3]] = values[:3]
        set_context(True, cu_seqlens_q=torch.tensor([0, 2]),
                    cu_seqlens_k=torch.tensor([0, 5]),
                    slot_mapping=slots[3:5], block_tables=table)
        torch.testing.assert_close(self.attn(q[3:5], k[3:5], v[3:5]),
                                   self.reference(q[3:5], k[:5], v[:5]))
        set_context(False, slot_mapping=slots[5:], context_lens=torch.tensor([6]),
                    block_tables=table)
        torch.testing.assert_close(self.attn(q[5:], k[5:], v[5:]),
                                   self.reference(q[5:], k, v))


if __name__ == "__main__":
    unittest.main()

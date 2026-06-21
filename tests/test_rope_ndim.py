"""Tests for RoPE application with different cos ndim (including mRoPE ndim=4)."""

import torch
from graspo.backends.native_tp.tensor_utils import _apply_rope, _apply_rope_partial


def _dummy_rope_cache(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal _rope_cache-like output: cos/sin of shape (seq_len, head_dim)."""
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _mrope_cos_sin(batch: int, seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate _qwen35_mrope_embeddings output: cos/sin of shape (1, B, S, head_dim)."""
    cos, sin = _dummy_rope_cache(seq_len, head_dim)
    # Add mrope batch dimension: (1, B, S, head_dim) where first dim is mrope dims=1
    return cos.unsqueeze(0).expand(1, batch, -1, -1).clone(), sin.unsqueeze(0).expand(1, batch, -1, -1).clone()


class TestRopeNdims:
    """Test _apply_rope and _apply_rope_partial handle all cos ndim cases."""

    B, H, S, D = 2, 8, 64, 128

    def _make_query_key(self, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        s = seq_len or self.S
        q = torch.randn(self.B, self.H, s, self.D)
        k = torch.randn(self.B, self.H, s, self.D)
        return q, k

    # ── _apply_rope_partial ──────────────────────────────────────────

    def test_partial_cos_ndim_2(self):
        """cos ndim=2: index by position_ids, then unsqueeze head dim."""
        cos, sin = _dummy_rope_cache(self.S, self.D)
        q, k = self._make_query_key()
        position_ids = torch.arange(self.S).unsqueeze(0).expand(self.B, -1)
        q_out, k_out = _apply_rope_partial(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_partial_cos_ndim_3(self):
        """cos ndim=3: unsqueeze head dim directly."""
        cos, sin = _dummy_rope_cache(self.S, self.D)
        cos = cos.unsqueeze(0).expand(self.B, -1, -1).clone()
        sin = sin.unsqueeze(0).expand(self.B, -1, -1).clone()
        assert cos.ndim == 3
        q, k = self._make_query_key()
        position_ids = torch.arange(self.S).unsqueeze(0).expand(self.B, -1)
        q_out, k_out = _apply_rope_partial(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_partial_cos_ndim_4_mrope(self):
        """cos ndim=4 (mRoPE path): unsqueeze head dim at position 2, no indexing."""
        cos, sin = _mrope_cos_sin(self.B, self.S, self.D)
        assert cos.ndim == 4
        q, k = self._make_query_key()
        # mRoPE position_ids: (3, B, S) — 3D temporal/height/width
        position_ids = torch.randint(0, 30, (3, self.B, self.S))
        q_out, k_out = _apply_rope_partial(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_partial_cos_ndim_4_decode_step(self):
        """mRoPE cos (ndim=4) during decode with growing sequence length.

        Simulates: prefill at S=925 produces mrope cos (1,B,925,D), then
        decode step with query_len=926 should NOT crash with dimension mismatch.
        """
        S_prefill = 925
        S_decode = S_prefill + 1
        head_dim = 64

        # Prefill-style mrope cos
        cos_mrope, sin_mrope = _mrope_cos_sin(1, S_prefill, head_dim)
        assert cos_mrope.ndim == 4
        assert cos_mrope.shape == (1, 1, S_prefill, head_dim)

        # Decode step: query grows by 1
        q_decode = torch.randn(1, 8, S_decode, head_dim)
        k_decode = torch.randn(1, 8, S_decode, head_dim)
        position_ids = torch.arange(S_decode).unsqueeze(0)  # ndim=2

        # If _rope_cache is used (ndim=2), this works fine
        cos_cache, sin_cache = _dummy_rope_cache(S_decode, head_dim)
        q_out, k_out = _apply_rope_partial(q_decode, k_decode, cos_cache, sin_cache, position_ids)
        assert q_out.shape == q_decode.shape

        # But the critical case: what if attention layer produces mrope-style
        # cos (ndim=4) during decode? Our fix must handle this.
        cos_mrope_decode, sin_mrope_decode = _mrope_cos_sin(1, S_decode, head_dim)
        assert cos_mrope_decode.ndim == 4
        assert cos_mrope_decode.shape == (1, 1, S_decode, head_dim)
        q_out, k_out = _apply_rope_partial(
            q_decode, k_decode, cos_mrope_decode, sin_mrope_decode, position_ids
        )
        assert q_out.shape == q_decode.shape
        assert k_out.shape == k_decode.shape

    # ── _apply_rope (non-partial variant) ────────────────────────────

    def test_rope_cos_ndim_2(self):
        cos, sin = _dummy_rope_cache(self.S, self.D)
        q, k = self._make_query_key()
        position_ids = torch.arange(self.S).unsqueeze(0).expand(self.B, -1)
        q_out, k_out = _apply_rope(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape

    def test_rope_cos_ndim_3(self):
        cos, sin = _dummy_rope_cache(self.S, self.D)
        cos = cos.unsqueeze(0).expand(self.B, -1, -1).clone()
        sin = sin.unsqueeze(0).expand(self.B, -1, -1).clone()
        assert cos.ndim == 3
        q, k = self._make_query_key()
        position_ids = torch.arange(self.S).unsqueeze(0).expand(self.B, -1)
        q_out, k_out = _apply_rope(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape

    def test_rope_cos_ndim_4_mrope(self):
        """_apply_rope must also handle ndim=4 (mRoPE)."""
        cos, sin = _mrope_cos_sin(self.B, self.S, self.D)
        assert cos.ndim == 4
        q, k = self._make_query_key()
        position_ids = torch.randint(0, 30, (3, self.B, self.S))
        q_out, k_out = _apply_rope(q, k, cos, sin, position_ids)
        assert q_out.shape == q.shape

    def test_rope_cos_ndim_4_decode_step(self):
        """_apply_rope mRoPE decode step parity with _apply_rope_partial."""
        S_decode = 926
        head_dim = 64
        B = 1
        cos_mrope, sin_mrope = _mrope_cos_sin(B, S_decode, head_dim)
        assert cos_mrope.ndim == 4
        q = torch.randn(B, 8, S_decode, head_dim)
        k = torch.randn(B, 8, S_decode, head_dim)
        position_ids = torch.arange(S_decode).unsqueeze(0)
        q_out, k_out = _apply_rope(q, k, cos_mrope, sin_mrope, position_ids)
        assert q_out.shape == q.shape

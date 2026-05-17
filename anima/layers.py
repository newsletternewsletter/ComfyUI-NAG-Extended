import torch
from einops import rearrange

from ..utils import nag


def nag_cross_attn_forward(
    self,
    x,
    context=None,
    rope_emb=None,
    transformer_options=None,
    context_pad_len: int = 0,
    nag_pad_len: int = 0,
):
    """
    Drop-in replacement for Cosmos / Anima `Block.cross_attn.forward` that
    applies Normalized Attention Guidance (NAG).

    The image latent batch (`x`) is kept at the positive batch size; only the
    text context is extended with the NAG-negative branch. Two attention calls
    are issued — positive (Q · K_pos / V_pos) and negative (same Q · K_neg /
    V_neg) — and their outputs are combined via the NAG formula.

    Mirrors `wan/model.py:NAGWanT2VCrossAttention.forward`, adapted to the
    Cosmos `Attention` module's attribute names (`q_proj`/`k_proj`/`v_proj`,
    `q_norm`/`k_norm`/`v_norm`, `attn_op`, `output_proj`, `output_dropout`).

    Args:
        x:       [pos_bsz, L_img, D]               — positive image latent
        context: [pos_bsz + nag_bsz, L_ctx, D_ctx] — pos+neg text context
    """
    if transformer_options is None:
        transformer_options = {}

    origin_bsz = len(context) - len(x)
    assert origin_bsz > 0, "nag_cross_attn_forward expects an extended context batch"

    # Q from positive image only; K, V from extended context.
    q = self.q_proj(x)
    k = self.k_proj(context)
    v = self.v_proj(context)

    # Split heads to match `Attention.compute_qkv` / `torch_attention_op`.
    q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
    k = rearrange(k, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
    v = rearrange(v, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)
    v = self.v_norm(v)
    # cross_attn: is_selfattn is False → no rope, matching the base impl.

    # Strip any sequence padding `cat_context` introduced and separate pos/neg K, V.
    k_pos = k[:-origin_bsz, context_pad_len:]
    k_neg = k[-origin_bsz:, nag_pad_len:]
    v_pos = v[:-origin_bsz, context_pad_len:]
    v_neg = v[-origin_bsz:, nag_pad_len:]

    # Reuse Q for the negative pass (image batch is identical on both sides).
    q_neg = q[-origin_bsz:]

    out_pos = self.attn_op(q,     k_pos, v_pos, transformer_options=transformer_options)
    out_neg = self.attn_op(q_neg, k_neg, v_neg, transformer_options=transformer_options)

    # NAG-combine the tail of the positive output (paired with the negative
    # output). Use clone + slice-assign (matching the SDXL port in
    # `sd/attention.py:118-121`) rather than `cat([out_pos[:-origin_bsz], ...])`
    # because at `pos_bsz == origin_bsz == 1` (the Anima Turbo case) the
    # left half would be an empty tensor and the cat path is needlessly fragile.
    out_guided = nag(out_pos[-origin_bsz:], out_neg, self.nag_scale, self.nag_tau, self.nag_alpha)
    out = out_pos.clone()
    out[-origin_bsz:] = out_guided

    if out.dtype == torch.float16:
        out = torch.nan_to_num(out, nan=0.0, posinf=65504, neginf=-65504)

    return self.output_dropout(self.output_proj(out))

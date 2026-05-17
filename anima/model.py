from functools import partial
from types import MethodType

import torch

from .layers import nag_cross_attn_forward
from ..utils import cat_context, check_nag_activation, NAGSwitch


def forward_nag_anima(
    self,
    x,
    timestep,
    context,
    y=None,

    nag_negative_context=None,
    nag_sigma_start=14.7,
    nag_sigma_end=0.0,

    **kwargs,
):
    """
    NAG wrapper for Anima / Cosmos forward.

    Only the cross-attention context is extended with the NAG-negative branch;
    `x` / `timestep` / `y` stay at the positive batch size, so `self_attn`,
    `mlp`, AdaLN, patch & final layers are NOT duplicated. The extra cost is
    limited to one extra cross-attention pass per block.

    `nag_negative_context` is expected to already be in the LLM-adapter output
    space (post-`preprocess_text_embeds`), matching the positive `context`.
    `NAGAnimaSwitch.set_nag` handles that conversion once at sampling start,
    mirroring what `Anima.extra_conds` does for the positive cond.
    """
    transformer_options = kwargs.get("transformer_options", {})

    apply_nag = check_nag_activation(transformer_options, nag_sigma_start, nag_sigma_end)
    if not apply_nag or nag_negative_context is None:
        return self.forward_orig_anima(x, timestep, context, y=y, **kwargs)

    origin_context_len = context.shape[1]
    context_extended = cat_context(context, nag_negative_context, trim_context=True)
    context_pad_len = context_extended.shape[1] - origin_context_len
    nag_pad_len = context_extended.shape[1] - nag_negative_context.shape[1]

    # Temporarily wrap each block's cross_attn with the NAG version.
    cross_attns_forward = []
    try:
        for block in self.blocks:
            if not hasattr(block, "cross_attn"):
                continue
            cross_attn = block.cross_attn
            cross_attns_forward.append((cross_attn, cross_attn.forward))
            cross_attn.forward = MethodType(
                partial(
                    nag_cross_attn_forward,
                    context_pad_len=context_pad_len,
                    nag_pad_len=nag_pad_len,
                ),
                cross_attn,
            )

        return self.forward_orig_anima(x, timestep, context_extended, y=y, **kwargs)
    finally:
        for cross_attn, forward_fn in cross_attns_forward:
            cross_attn.forward = forward_fn


def _adapt_nag_negative_context(model, nag_negative_cond):
    """
    Mirror what `Anima.extra_conds` (`ComfyUI/comfy/model_base.py:1219-1238`)
    does for the positive cond: run the raw Qwen3-0.6B hidden states through
    `LLMAdapter` (via `preprocess_text_embeds`) using the `t5xxl_ids` /
    `t5xxl_weights` the Anima text encoder packed into the cond's extras dict
    (`ComfyUI/comfy/text_encoders/anima.py:50-51`).

    This is the missing step that caused NAG to silently no-op: without it,
    the positive context arrives at `cross_attn` already adapter-projected
    while the negative context is raw Qwen3 — `k_proj(raw_qwen)` then produces
    noise instead of meaningful K, and the NAG combine collapses back to
    `z_positive`.

    Returns the adapted negative context, or the input unchanged + a warning
    if no `t5xxl_ids` is available (e.g. the user wired a non-Anima text
    encoder to the NAG-negative input).
    """
    raw = nag_negative_cond[0][0]
    extras = nag_negative_cond[0][1] if len(nag_negative_cond[0]) > 1 and isinstance(nag_negative_cond[0][1], dict) else {}

    t5xxl_ids = extras.get("t5xxl_ids")
    t5xxl_weights = extras.get("t5xxl_weights")

    if t5xxl_ids is None:
        print(
            "[NAG-Anima] WARNING: the NAG-negative conditioning has no "
            "`t5xxl_ids` — it was not encoded with the Anima text encoder. "
            "NAG will likely have no visible effect because the negative "
            "context is in the wrong feature space for the model's "
            "cross-attention. Wire the Anima CLIP into the NAG-negative input."
        )
        return raw

    # Use the LLM adapter's own parameter dtype, not `model.dtype` /
    # `next(model.parameters()).dtype`. The diffusion blocks can be loaded at
    # a different precision than the adapter (e.g. main weights fp16 while the
    # adapter is stored bf16) and `comfy.ops.Linear.forward` only auto-casts
    # weights when `comfy_cast_weights` is active — otherwise the matmul
    # requires the input dtype to match the stored weight dtype exactly.
    adapter_param = next(model.llm_adapter.parameters())
    device = adapter_param.device
    dtype = adapter_param.dtype

    # Match `Anima.extra_conds`' reshape exactly.
    t5xxl_ids = t5xxl_ids.unsqueeze(0) if t5xxl_ids.ndim == 1 else t5xxl_ids
    if t5xxl_weights is not None:
        if t5xxl_weights.ndim == 1:
            t5xxl_weights = t5xxl_weights.unsqueeze(0).unsqueeze(-1)
        t5xxl_weights = t5xxl_weights.to(device=device, dtype=dtype)

    adapted = model.preprocess_text_embeds(
        raw.to(device=device, dtype=dtype),
        t5xxl_ids.to(device=device),
        t5xxl_weights=t5xxl_weights,
    )
    return adapted


class NAGAnimaSwitch(NAGSwitch):
    """
    Switcher for enabling / disabling NAG on Anima / Cosmos models.

    NAG is applied at the cross-attention layer (where the text context flows
    in), matching the pattern used by SDXL (`sd/attention.py`) and the
    reference implementation. The image-latent batch is never doubled, keeping
    the per-step cost increase small — important for the distilled Anima Turbo
    variants that run at CFG=1.

    Before the actual sampling loop starts, we run the NAG-negative context
    through the same `LLMAdapter` that `Anima.extra_conds` applies to the
    positive cond. Skipping that adapter is what made all previous NAG-for-
    Anima implementations silently produce no negative-prompt effect.
    """

    def set_nag(self):
        # 0. Bring the NAG-negative context into the same feature space the
        #    cross-attention expects (LLM-adapter output, [B, 512, 1024]).
        adapted_neg = _adapt_nag_negative_context(self.model, self.nag_negative_cond)

        # 1. Wrap the model forward so we can intercept and extend context for NAG.
        if getattr(self.model, "forward", None) is not None and not hasattr(self.model, "forward_orig_anima"):
            self.model.forward_orig_anima = self.model.forward

        self.model.forward = MethodType(
            partial(
                forward_nag_anima,
                nag_negative_context=adapted_neg,
                nag_sigma_start=self.nag_sigma_start,
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )

        # 2. Attach NAG hyperparameters to each cross_attn so the patched forward
        #    can read them. (cross_attn.forward itself is wrapped lazily inside
        #    forward_nag_anima and restored every step via try/finally.)
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                if hasattr(block, "cross_attn"):
                    block.cross_attn.nag_scale = self.nag_scale
                    block.cross_attn.nag_tau = self.nag_tau
                    block.cross_attn.nag_alpha = self.nag_alpha

    def set_origin(self):
        """Restore everything back to normal when sampling finishes."""
        # 1. Restore the main forward.
        if hasattr(self.model, "forward_orig_anima"):
            self.model.forward = self.model.forward_orig_anima
            delattr(self.model, "forward_orig_anima")

        # 2. Clean up attached NAG attributes on each cross_attn.
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                if hasattr(block, "cross_attn"):
                    for attr in ("nag_scale", "nag_tau", "nag_alpha"):
                        if hasattr(block.cross_attn, attr):
                            delattr(block.cross_attn, attr)

from typing import Set

import torch.nn as nn

from clipora.lora.attention import InjectedMultiHeadAttention


def inject_linear_attention(
    model: nn.Module,
    encoders: Set[str] = {"transformer", "visual"},
    embed_dim: int = 768,
    num_heads: int = 12,
):
    for encoder in encoders:
        sub_modules = encoder.split(".")
        target_module = model
        for sub_module in sub_modules:
            target_module = getattr(target_module, sub_module, None)
            if target_module is None:
                break

        if target_module is not None and hasattr(target_module, "resblocks"):
            for module in target_module.resblocks:
                injection = InjectedMultiHeadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                injection.set_parameters(module.attn)
                module.attn = injection
    return model

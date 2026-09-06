"""LoRA (Low-Rank Adaptation) wrapper for DOFA's ViT attention layers.

Freezes the base Linear weights and injects small trainable low-rank
update matrices (A, B) alongside them, following Hu et al. 2021.
Applied to the qkv and proj Linear layers inside each transformer block.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a frozen base + trainable low-rank update.
    output = base_linear(x) + (alpha / r) * B(A(x))
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero-initialized so the LoRA path starts as a no-op

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def apply_lora_to_dofa_backbone(backbone, r: int = 8, alpha: int = 16):
    """Freezes the DOFA backbone entirely, then injects LoRA adapters into
    each transformer block's qkv and proj Linear layers, which become the
    only trainable parameters inside the backbone.
    """
    for p in backbone.body.parameters():
        p.requires_grad = False

    for block in backbone.body.blocks:
        attn = block.attn
        attn.qkv = LoRALinear(attn.qkv, r=r, alpha=alpha)
        attn.proj = LoRALinear(attn.proj, r=r, alpha=alpha)

    return backbone
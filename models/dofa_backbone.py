"""DOFA ViT backbone wrapped for torchvision detection models.

Loads the pretrained DOFA ViT-Base, exposes its per-patch spatial features
(not the globally-pooled classification vector), and builds a 4-level
feature pyramid from that single-scale feature map using the ViTDet
"simple feature pyramid" approach (strided conv/deconv), since a plain ViT
has no native multi-scale hierarchy the way a CNN backbone does.
"""
import sys
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

DOFA_REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "third_party", "DOFA")
sys.path.insert(0, os.path.abspath(DOFA_REPO_PATH))

from dofa_v1 import vit_base_patch16          # noqa: E402
from pretraining.util.pos_embed import interpolate_pos_embed  # noqa: E402

RGB_WAVELENGTHS = [0.665, 0.56, 0.49]  # micrometers; DOFA's own RGB/NAIP convention
DOFA_PATCH_SIZE = 16
DOFA_EMBED_DIM = 768  # ViT-Base


class DOFAFeatureExtractor(nn.Module):
    """Wraps DOFA's OFAViT to return the spatial (pre-pool) patch grid as a
    [B, C, H, W] feature map instead of a pooled classification vector.
    """

    def __init__(self, checkpoint_path: str, img_size: int = 800):
        super().__init__()
        self.img_size = img_size
        self.patch_size = DOFA_PATCH_SIZE
        self.grid_size = img_size // self.patch_size
        num_patches = self.grid_size ** 2

        # Build with the target num_patches directly so pos_embed is the
        # right final shape; we still need to interpolate the *checkpoint's*
        # pos_embed (native 224/patch16 = 14x14 = 196 patches) into this shape.
        model = vit_base_patch16(num_classes=0, global_pool=True)
        model.num_patches = num_patches
        model.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, DOFA_EMBED_DIM), requires_grad=False
        )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        interpolate_pos_embed(model, state_dict, num_patches=num_patches)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[DOFA] missing keys: {missing}")
        print(f"[DOFA] unexpected keys: {unexpected}")

        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.blocks = model.blocks

        self.register_buffer(
            "wavelengths", torch.tensor(RGB_WAVELENGTHS, dtype=torch.float32)
        )

    def forward(self, x):
        B = x.shape[0]
        x, _ = self.patch_embed(x, self.wavelengths.to(x.device))
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        x = x[:, 1:, :]  # drop cls token, keep per-patch tokens
        x = x.transpose(1, 2).reshape(B, DOFA_EMBED_DIM, self.grid_size, self.grid_size)
        return x  # [B, 768, grid, grid] -- stride = patch_size (16)


class DOFASimpleFPNBackbone(nn.Module):
    """ViTDet-style simple feature pyramid on top of DOFA's single-scale
    feature map. Produces 4 levels at strides {4, 8, 16, 32} relative to
    the input image, matching what torchvision's FasterRCNN + FPN expects.
    """

    out_channels = 256

    def __init__(self, checkpoint_path: str, img_size: int = 800, freeze_backbone: bool = False):
        super().__init__()
        self.body = DOFAFeatureExtractor(checkpoint_path, img_size=img_size)

        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad = False

        C = DOFA_EMBED_DIM
        O = self.out_channels

        # From stride-16 base feature map, produce stride-4, 8, 16, 32 levels.
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(C // 2, C // 4, kernel_size=2, stride=2),
        )  # stride 16 -> stride 4
        self.fpn2 = nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2)  # stride 16 -> 8
        self.fpn3 = nn.Identity()  # stride 16 stays 16
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)  # stride 16 -> 32

        self.lateral = nn.ModuleDict({
            "p2": nn.Conv2d(C // 4, O, kernel_size=1),
            "p3": nn.Conv2d(C // 2, O, kernel_size=1),
            "p4": nn.Conv2d(C, O, kernel_size=1),
            "p5": nn.Conv2d(C, O, kernel_size=1),
        })
        self.smooth = nn.ModuleDict({
            k: nn.Conv2d(O, O, kernel_size=3, padding=1) for k in ["p2", "p3", "p4", "p5"]
        })

    def forward(self, x):
        feat = self.body(x)  # [B, 768, H/16, W/16]

        p2 = self.smooth["p2"](self.lateral["p2"](self.fpn1(feat)))
        p3 = self.smooth["p3"](self.lateral["p3"](self.fpn2(feat)))
        p4 = self.smooth["p4"](self.lateral["p4"](self.fpn3(feat)))
        p5 = self.smooth["p5"](self.lateral["p5"](self.fpn4(feat)))

        return OrderedDict([("0", p2), ("1", p3), ("2", p4), ("3", p5)])
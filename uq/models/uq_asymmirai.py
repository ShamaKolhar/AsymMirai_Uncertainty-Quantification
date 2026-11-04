"""UQ-ready, device-agnostic re-implementation of AsymMirai's LocalizedDifModel forward pass.

Numerically identical to the original model when dropout is off (see tests/test_model.py),
but adds:
  * PairedDropout: the SAME channel mask is used for the left and right breast of a view pair.
    Independent masks (what nn.Dropout on each embedding does) create artificial asymmetry
    and therefore fake "uncertainty".
  * Norm-preserving scaling in the head (1/sqrt(1-p)) so the expected squared asymmetry is
    unchanged by dropout -> MC means are not biased upwards.
  * Explicit MC mode (`model.mc(True)`) instead of toggling .train(), so BatchNorm layers in
    the backbone always stay in eval mode.
  * Pluggable backbone: the Mirai image encoder (released weights), torchvision ResNet-18, or
    ConvNeXt-Tiny (Phase-4 extension). Non-Mirai backbones need the head re-fitted
    (uq.finetune_head) before their predictions mean anything.
  * Returns localisation (argmax window per view) for localisation-uncertainty analysis.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ASYM_DIR = Path(__file__).resolve().parents[2] / "AsymMirai" / "asymmetry_model"
if str(ASYM_DIR) not in sys.path:
    sys.path.insert(0, str(ASYM_DIR))
from asymmetry_metrics import hybrid_asymmetry  # noqa: E402  (original AsymMirai metric)


class PairedDropout(nn.Module):
    """Channel dropout whose mask can be reused for the second image of a pair.

    scale='inverted' -> 1/(1-p)      (standard; use inside the backbone)
    scale='norm'     -> 1/sqrt(1-p)  (keeps E[||x||^2]; use on the embeddings before the L2 asymmetry)
    """

    def __init__(self, p: float, scale: str = "norm"):
        super().__init__()
        self.p = float(p)
        self.scale = scale
        self.active = False
        self.reuse = False
        self._mask = None

    def forward(self, x):
        if not self.active or self.p <= 0:
            return x
        if not (self.reuse and self._mask is not None and self._mask.shape[:2] == x.shape[:2]):
            keep = 1.0 - self.p
            self._mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1, 1), keep, device=x.device))
            self._mask = self._mask / (keep if self.scale == "inverted" else math.sqrt(keep))
        return x * self._mask


def insert_backbone_dropout(seq: nn.Sequential, p: float) -> nn.Sequential:
    """Insert PairedDropout after every residual stage (named layer*/stages)."""
    mods = []
    for name, m in seq.named_children():
        mods.append(m)
        if name.startswith("layer") or name.isdigit():
            mods.append(PairedDropout(p, scale="inverted"))
    return nn.Sequential(*mods)


# ----------------------------------------------------------------------------- backbones
def build_backbone(kind: str, pretrained: bool = True, mirai_snapshot: str | None = None) -> tuple[nn.Module, int]:
    """Return (feature extractor, output channels)."""
    import torchvision.models as tvm

    if kind == "mirai":
        if mirai_snapshot is None:
            raise ValueError("Mirai backbone requires paths.weights.mirai_snapshot")
        from uq.models.loading import extract_mirai_backbone
        return extract_mirai_backbone(mirai_snapshot), 512
    if kind == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        return nn.Sequential(*list(m.children())[:-2]), 512
    if kind == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        return m.features, 768
    raise ValueError(f"unknown backbone {kind}")


# ----------------------------------------------------------------------------- model
class UQAsymMirai(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_channel: int = 512,
                 latent_h: int = 5, latent_w: int = 5, flexible_asymmetry: bool = True,
                 use_stretch: bool = True, use_stretch_matrix: bool = True,
                 asym_mean: float = 2000.0, asym_std: float = 300.0,
                 head_dropout: float = 0.0, backbone_dropout: float = 0.0,
                 alignment_space: str | None = None, use_bn: bool = False,
                 topk_for_heatmap: int | None = None, backbone_name: str = "mirai"):
        super().__init__()
        if backbone_dropout > 0:
            backbone = insert_backbone_dropout(backbone, backbone_dropout)
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.embedding_channel = embedding_channel
        self.latent_h, self.latent_w = latent_h, latent_w
        self.flexible_asymmetry = flexible_asymmetry
        self.use_stretch, self.use_stretch_matrix = use_stretch, use_stretch_matrix
        self.alignment_space = alignment_space
        self.use_bn = use_bn
        self.topk_for_heatmap = topk_for_heatmap
        eye = torch.eye(embedding_channel) if use_stretch_matrix else torch.ones(embedding_channel)
        self.cc_stretch_params = nn.Parameter(eye.clone())
        self.mlo_stretch_params = nn.Parameter(eye.clone())
        self.asym_mean = nn.Parameter(torch.tensor(float(asym_mean)))
        self.asym_std = nn.Parameter(torch.tensor(float(asym_std)))
        self.bn = nn.BatchNorm2d(embedding_channel) if use_bn else None
        self.topk_weights = (nn.Parameter(torch.arange(topk_for_heatmap).float() / torch.arange(topk_for_heatmap).sum())
                             if topk_for_heatmap else None)
        self.head_dropout = PairedDropout(head_dropout, scale="norm")
        self.hparams = dict(embedding_channel=embedding_channel, latent_h=latent_h, latent_w=latent_w,
                            flexible_asymmetry=flexible_asymmetry, use_stretch=use_stretch,
                            use_stretch_matrix=use_stretch_matrix, head_dropout=head_dropout,
                            backbone_dropout=backbone_dropout, alignment_space=alignment_space,
                            use_bn=use_bn, topk_for_heatmap=topk_for_heatmap, backbone_name=backbone_name)

    # ---- MC switches
    def _dropouts(self):
        return [m for m in self.modules() if isinstance(m, PairedDropout)]

    def mc(self, on: bool = True) -> "UQAsymMirai":
        self.eval()  # BN always in eval mode
        for m in self._dropouts():
            m.active = on
        return self

    @property
    def has_stochastic_layers(self) -> bool:
        return any(m.p > 0 for m in self._dropouts())

    def _set_reuse(self, reuse: bool):
        for m in self._dropouts():
            m.reuse = reuse

    # ---- forward pieces
    def _stretch(self, emb, view):
        if not self.use_stretch:
            return emb
        W = self.cc_stretch_params if view == "CC" else self.mlo_stretch_params
        if self.use_stretch_matrix:
            b, c, h, w = emb.shape
            return (W @ emb.transpose(0, 1).reshape(c, -1)).view(c, b, h, w).transpose(0, 1)
        return W.view(1, -1, 1, 1) * emb

    def features(self, left_cc, right_cc, left_mlo, right_mlo) -> dict:
        """Backbone embeddings per view pair. Deterministic unless backbone dropout is active."""
        feats = {}
        for left, right, view in [(left_cc, right_cc, "CC"), (left_mlo, right_mlo, "MLO")]:
            missing = (left.sum(dim=(1, 2, 3)) == 0).float()
            if self.alignment_space == "pixel":
                right = torch.flip(right, dims=[-1])
            self._set_reuse(False)
            le = self.backbone(left)
            self._set_reuse(True)          # same backbone-dropout masks for the right breast
            re = self.backbone(right)
            self._set_reuse(False)
            if self.alignment_space == "latent":
                re = torch.flip(re, dims=[-1])
            feats[view] = (le, re, missing)
        return feats

    def forward(self, left_cc, right_cc, left_mlo, right_mlo):
        return self.head(self.features(left_cc, right_cc, left_mlo, right_mlo))

    def head(self, feats: dict) -> dict:
        """Everything after the backbone. Can be re-run on cached features for cheap MC dropout."""
        some = feats["CC"][0]
        dev, B = some.device, some.shape[0]
        count = torch.full((B,), 2.0, device=dev)
        total = torch.zeros(B, device=dev)
        loc = {}
        for view in ("CC", "MLO"):
            le, re, missing = feats[view]
            count = count - missing
            self.head_dropout.reuse = False
            le = self.head_dropout(le)
            self.head_dropout.reuse = True   # same mask for the right breast
            re = self.head_dropout(re)
            self.head_dropout.reuse = False
            le, re = self._stretch(le, view), self._stretch(re, view)
            if self.bn is not None:
                le, re = self.bn(le), self.bn(re)
            asym, other = hybrid_asymmetry(le, re, latent_h=self.latent_h, latent_w=self.latent_w,
                                           flexible=self.flexible_asymmetry, topk=self.topk_for_heatmap)
            if self.topk_weights is not None:
                asym = (asym @ self.topk_weights.view(-1, 1)).view(-1)
            total = total + asym * (1 - missing)
            loc[view] = self._decode_location(other)
        raw = total / count.clamp(min=1)
        score = raw if self.bn is not None else (raw - self.asym_mean) / self.asym_std
        prob = torch.sigmoid(score)
        return {"prob": prob, "logit": score, "raw_asymmetry": raw,
                "cc_y": loc["CC"]["y_argmin"], "cc_x": loc["CC"]["x_argmin"],
                "mlo_y": loc["MLO"]["y_argmin"], "mlo_x": loc["MLO"]["x_argmin"],
                "cc_heatmap": loc["CC"]["heatmap"], "mlo_heatmap": loc["MLO"]["heatmap"]}

    @staticmethod
    def _decode_location(other: dict) -> dict:
        """hybrid_asymmetry returns y_argmin = column index of the max for EVERY row (B, H') and
        x_argmin = the winning row (B,). Convert to a proper (row, col) of the most asymmetric window."""
        row = other["x_argmin"]
        cols = other["y_argmin"]
        if torch.is_tensor(row) and torch.is_tensor(cols) and cols.dim() == 2:
            col = cols.gather(1, row.view(-1, 1)).view(-1)
        else:  # topk mode returns -1 placeholders
            row = col = torch.full((other["heatmap"].shape[0],), -1, dtype=torch.long, device=other["heatmap"].device)
        return {"y_argmin": row, "x_argmin": col, "heatmap": other["heatmap"]}

    # ---- head-only parameters (what AsymMirai trains)
    HEAD_KEYS = ("cc_stretch_params", "mlo_stretch_params", "asym_mean", "asym_std", "bn.", "topk_weights")

    def head_state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith(self.HEAD_KEYS)}

    def load_head_state(self, sd: dict):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected

    @property
    def backbone_is_stochastic(self) -> bool:
        return any(m.p > 0 for m in self.backbone.modules() if isinstance(m, PairedDropout))

    def head_parameters(self):
        ps = [self.cc_stretch_params, self.mlo_stretch_params, self.asym_mean, self.asym_std]
        if self.bn is not None:
            ps += list(self.bn.parameters())
        if self.topk_weights is not None:
            ps.append(self.topk_weights)
        return ps

"""Loading trained AsymMirai weights into the UQ model.

The released AsymMirai checkpoint (Duke box link in AsymMirai/README.md) is a *fully pickled*
`LocalizedDifModel` saved with `torch.save(model, ...)`. Unpickling it needs the ORIGINAL
class definition, which is why the original AsymMirai code is restored unmodified in ./AsymMirai
(the old project had overwritten `mirai_localized_dif_head.py`, so the released weights could
not be loaded, and the model silently ran with an untrained head on an ImageNet ResNet-18).

We read the trained pieces off the pickled object (Mirai backbone, CC/MLO stretch matrices,
learned asymmetry mean/std, pooling settings) and copy them into `UQAsymMirai`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from uq.models.uq_asymmirai import UQAsymMirai, build_backbone

ASYM_ROOT = Path(__file__).resolve().parents[2] / "AsymMirai"


def _ensure_original_code_on_path():
    for p in (ASYM_ROOT, ASYM_ROOT / "asymmetry_model"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _torch_load(path, map_location="cpu"):
    _ensure_original_code_on_path()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location=map_location)


def extract_mirai_backbone(snapshot_path: str, cutoff_layer: str = "layer4_1") -> nn.Sequential:
    """Same logic as the original AsymMirai extract_mirai_backbone, but device-agnostic."""
    mirai = _torch_load(snapshot_path)
    if isinstance(mirai, nn.DataParallel):
        mirai = mirai.module
    layers = []
    for top in mirai.children():
        for m in top.children():
            for name, n in m.named_children():
                layers.append(n)
                if name == cutoff_layer:
                    break
            break
        break
    return nn.Sequential(*layers)


def _scalar(x, default):
    if x is None:
        return float(default)
    return float(x.detach().cpu().item() if torch.is_tensor(x) else x)


def from_original_model(orig: nn.Module, head_dropout: float = 0.0, backbone_dropout: float = 0.0) -> UQAsymMirai:
    """Copy a trained original LocalizedDifModel into UQAsymMirai."""
    backbone = orig.backbone.module if isinstance(orig.backbone, nn.DataParallel) else orig.backbone
    use_stretch_matrix = bool(getattr(orig, "use_stretch_matrix", False))
    cc = orig.cc_stretch_params.detach().cpu()
    channels = cc.shape[0]
    # Original forward uses learned_asym_mean/std; fall back to initial_* if absent
    mean = _scalar(getattr(orig, "learned_asym_mean", None), getattr(orig, "initial_asym_mean", 2000))
    std = _scalar(getattr(orig, "learned_asym_std", None), getattr(orig, "initial_asym_std", 300))
    use_bn = bool(getattr(orig, "use_bn", False))
    topk = getattr(orig, "topk_for_heatmap", None)
    m = UQAsymMirai(backbone, embedding_channel=channels,
                    latent_h=getattr(orig, "latent_h", 5), latent_w=getattr(orig, "latent_w", 5),
                    flexible_asymmetry=getattr(orig, "flexible_asymmetry", True),
                    use_stretch=getattr(orig, "use_stretch", True), use_stretch_matrix=use_stretch_matrix,
                    asym_mean=mean, asym_std=std, head_dropout=head_dropout,
                    backbone_dropout=backbone_dropout,
                    alignment_space=getattr(orig, "alignment_space", None), use_bn=use_bn,
                    topk_for_heatmap=topk, backbone_name="mirai")
    with torch.no_grad():
        m.cc_stretch_params.copy_(cc)
        m.mlo_stretch_params.copy_(orig.mlo_stretch_params.detach().cpu())
        if use_bn and getattr(orig, "bn", None) is not None:
            m.bn.load_state_dict(orig.bn.state_dict())
        if topk and getattr(orig, "topk_weights", None) is not None:
            m.topk_weights.copy_(orig.topk_weights.detach().cpu())
    return m


def save_checkpoint(model: UQAsymMirai, path, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, backbone: nn.Module | None = None, mirai_snapshot=None, pretrained=False) -> UQAsymMirai:
    ck = _torch_load(path)
    hp = dict(ck["hparams"])
    if backbone is None:
        backbone, _ = build_backbone(hp["backbone_name"], pretrained=pretrained, mirai_snapshot=mirai_snapshot)
    m = UQAsymMirai(backbone, **hp)
    m.load_state_dict(ck["state_dict"])
    return m


def load_model(cfg, checkpoint: str | None = None, device=None) -> UQAsymMirai:
    """Build the UQ model described by the config.

    Priority: explicit `checkpoint` (our own format) > released AsymMirai full model > fresh backbone.
    A fresh (untrained-head) model is only allowed for non-Mirai backbones and prints a loud warning.
    """
    mcfg = cfg["model"]
    hd, bd = mcfg.get("head_dropout", 0.0), mcfg.get("backbone_dropout", 0.0)
    if checkpoint:
        obj = _torch_load(checkpoint)
        if isinstance(obj, dict) and "hparams" in obj:
            m = load_checkpoint(checkpoint, mirai_snapshot=_maybe(cfg, "weights", "mirai_snapshot"))
            m.head_dropout.p = hd
        else:
            m = from_original_model(obj, hd, bd)
    elif mcfg["backbone"] == "mirai":
        full = cfg.path("weights", "asymmirai_full_model")
        if not full.exists():
            raise FileNotFoundError(
                f"Trained AsymMirai weights not found at {full}. Download them from the link in "
                "AsymMirai/README.md ('Our trained AsymMirai weights are available here'). "
                "Running without them gives a constant prediction (the bug in the old results)."
            )
        m = from_original_model(_torch_load(full), hd, bd)
    else:
        backbone, ch = build_backbone(mcfg["backbone"], pretrained=True)
        m = UQAsymMirai(backbone, embedding_channel=ch, head_dropout=hd, backbone_dropout=bd,
                        alignment_space=mcfg.get("alignment_space"), backbone_name=mcfg["backbone"])
        print(f"WARNING: {mcfg['backbone']} backbone with an UNTRAINED head. Run uq.finetune_head "
              "first and pass the resulting checkpoint; otherwise predictions are not meaningful.")
    return m.to(device) if device is not None else m


def _maybe(cfg, *keys):
    try:
        p = cfg.path(*keys)
        return str(p) if p.exists() else None
    except Exception:
        return None

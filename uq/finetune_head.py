"""Fine-tune the AsymMirai head (CC/MLO stretch matrices + asymmetry mean/std) with dropout ON.

Why: MC Dropout only approximates Bayesian uncertainty if the network was *trained* with
that dropout. The released AsymMirai weights were trained without it, so we re-fit the small
head (the backbone stays frozen, exactly like AsymMirai's own training) with paired head dropout.

The same script:
  * builds deep-ensemble members  (--seeds 0 1 2 3 4: different init + bootstrap per seed)
  * fits the head for a new backbone (model.backbone: resnet18 | convnext_tiny) - Phase-4 extension.
    For new backbones asym_mean/std are first initialised from the data so the sigmoid isn't saturated
    (with an unfitted offset every exam would get the same saturated prediction).

Training uses the `finetune` split (set split.finetune_fraction > 0 in the config and rebuild
the metadata), so the calib/test patients are never seen.

Usage:
    python -m uq.finetune_head --config configs/uq_config.yaml --epochs 5 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uq.config import get_device, load_config, set_seed
from uq.dataset import VIEW_KEYS, ExamDataset, exams_wide
from uq.models.loading import load_model, save_checkpoint
from uq.preprocessing import Preprocessor


@torch.no_grad()
def init_asym_stats(model, loader, device, max_batches: int = 50):
    model.mc(False)
    raws = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        x = [b[k].to(device) for k in VIEW_KEYS]
        raws.append(model(*x)["raw_asymmetry"].cpu())
    r = torch.cat(raws)
    model.asym_mean.data.fill_(float(r.mean()))
    model.asym_std.data.fill_(float(r.std().clamp(min=1e-6)))
    print(f"Initialised asym_mean={model.asym_mean.item():.4g}, asym_std={model.asym_std.item():.4g}")


def train_one(cfg, seed: int, epochs: int, lr: float, horizon: float, device, out_dir: Path,
              checkpoint: str | None, bootstrap: bool) -> Path:
    set_seed(seed)
    exams = exams_wide(pd.read_csv(cfg.path("exam_csv")))
    tr = exams[exams["split"] == "finetune"]
    if len(tr) == 0:
        raise ValueError("No 'finetune' exams. Set split.finetune_fraction > 0 and re-run uq.data.build_metadata.")
    if bootstrap:
        tr = tr.sample(frac=1.0, replace=True, random_state=seed)
    pre = Preprocessor.from_config(cfg)
    loader = DataLoader(ExamDataset(tr, pre, seed=seed), batch_size=cfg["uq"]["batch_size"],
                        shuffle=True, num_workers=cfg["uq"]["num_workers"])

    model = load_model(cfg, checkpoint).to(device)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    if model.backbone_name != "mirai" and checkpoint is None:
        init_asym_stats(model, loader, device)
    with torch.no_grad():  # per-seed perturbation for ensemble diversity (as in AsymMirai's own init)
        for W in (model.cc_stretch_params, model.mlo_stretch_params):
            W.add_(torch.randn_like(W) * 0.01)

    y_all = (tr["years_to_cancer"] <= horizon).astype(float).to_numpy()
    prev = y_all.mean()
    if prev in (0.0, 1.0):
        print(f"WARNING: fine-tune split has only one class (prevalence {prev:.0%}); using unweighted loss. "
              "Increase split.finetune_fraction.")
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor((1 - prev) / prev, device=device)
    opt = torch.optim.Adam(model.head_parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(epochs):
        model.mc(True)  # dropout ON while training -> consistent with MC Dropout at test time
        tot = n = 0
        for b in tqdm(loader, desc=f"seed {seed} epoch {ep}"):
            x = [b[k].to(device) for k in VIEW_KEYS]
            y = (b["years_to_cancer"] <= horizon).float().to(device)
            out = model(*x)
            loss = loss_fn(out["logit"], y)
            opt.zero_grad(); loss.backward(); opt.step()
            model.asym_std.data.clamp_(min=1e-6)
            tot += loss.item() * len(y); n += len(y)
        print(f"seed {seed} epoch {ep}: loss {tot / n:.4f}")
    path = out_dir / f"head_seed{seed}.pt"
    save_checkpoint(model, path, extra={"seed": seed, "horizon": horizon, "epochs": epochs, "lr": lr})
    ck = torch.load(path, weights_only=False)
    ck["head_state"] = model.head_state_dict()
    torch.save(ck, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=float, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg.path("outputs") / "finetuned_heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [train_one(cfg, s, args.epochs, args.lr, args.horizon, get_device(), out_dir, args.checkpoint,
                       bootstrap=not args.no_bootstrap and len(args.seeds) > 1) for s in args.seeds]
    print("Saved heads:\n  " + "\n  ".join(map(str, paths)))
    if len(paths) > 1:
        print("Add these to paths.weights.ensemble_members and 'ensemble' to uq.methods.")


if __name__ == "__main__":
    main()

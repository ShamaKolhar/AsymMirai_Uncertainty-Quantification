"""Run uncertainty quantification over an exam CSV.

Methods (configurable in uq.methods):
  deterministic  - one pass, dropout off, no TTA (always computed; the reference prediction)
  mc_dropout     - K passes with paired head dropout. The backbone is run ONCE per exam and the
                   dropout samples are drawn on the cached embeddings (~K x cheaper than re-running the backbone).
  tta            - K passes with paired (same for left/right) small affine + contrast jitter
  mc_tta         - K passes with both
  ensemble       - one pass per ensemble member head (paths.weights.ensemble_members)

Conditions (uq.conditions): `clean` first, then each corruption as its own condition, so
robustness is measured as a separate experiment.

Outputs in <outputs>/<run_name>/:
  samples_<condition>.npz       raw per-sample probabilities for every method (for any later metric)
  predictions_<condition>.csv   per-exam summary: mean, std, entropy, mutual information,
                                localisation dispersion, labels, subgroup columns
  run_info.json                 config snapshot, sanity checks, timings

Usage:
    python -m uq.inference --config configs/uq_config.yaml [--checkpoint path] [--split test] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uq.config import get_device, load_config, set_seed
from uq.dataset import VIEW_KEYS, ExamDataset, exams_wide
from uq.models.loading import _torch_load, load_model
from uq.preprocessing import PairedTTA, Preprocessor

EPS = 1e-12


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def summarise_samples(samples: np.ndarray) -> dict:
    """samples: (K, B) probabilities -> per-exam uncertainty measures."""
    mean = samples.mean(0)
    logit = np.log(np.clip(samples, EPS, 1)) - np.log(np.clip(1 - samples, EPS, 1))
    total = binary_entropy(mean)                       # predictive entropy (total uncertainty)
    aleatoric = binary_entropy(samples).mean(0)        # expected entropy
    return {"mean": mean, "std": samples.std(0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(mean),
            "logit_std": logit.std(0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(mean),
            "entropy": total, "mutual_info": np.maximum(total - aleatoric, 0.0)}


def localisation_dispersion(locs: np.ndarray) -> np.ndarray:
    """locs: (K, B, 4) argmax window coords (cc_y, cc_x, mlo_y, mlo_x).
    Returns mean over views of the fraction of samples that disagree with the modal window."""
    K, B, _ = locs.shape
    out = np.zeros(B)
    for b in range(B):
        fr = []
        for v in (slice(0, 2), slice(2, 4)):
            pts = [tuple(r) for r in locs[:, b, v]]
            vals, counts = np.unique(np.array(pts), axis=0, return_counts=True)
            fr.append(1 - counts.max() / K)
        out[b] = float(np.mean(fr))
    return out


def _loc_array(out) -> np.ndarray:
    return torch.stack([out["cc_y"], out["cc_x"], out["mlo_y"], out["mlo_x"]], 1).cpu().numpy()


def load_ensemble_heads(cfg) -> list[dict]:
    heads = []
    for p in cfg["paths"]["weights"].get("ensemble_members") or []:
        p = Path(p) if Path(p).is_absolute() else cfg.root_path / p
        ck = _torch_load(p)
        heads.append(ck.get("head_state") or {k: v for k, v in ck["state_dict"].items()
                                               if not k.startswith("backbone.")})
    return heads


@torch.no_grad()
def run_batch(model, batch, methods, cfg, tta: PairedTTA, gen: torch.Generator, device, ensemble_heads,
              bg_value: float) -> dict:
    x = [batch[k].to(device, non_blocking=True) for k in VIEW_KEYS]
    K_mc, K_tta = cfg["uq"]["mc_samples"], cfg["uq"]["tta_samples"]
    res = {}

    model.mc(False)
    feats = model.features(*x)
    det = model.head(feats)
    res["deterministic"] = (det["prob"].cpu().numpy()[None], _loc_array(det)[None])

    def tta_inputs():
        lcc, rcc = tta(x[0], x[1], gen, bg_value)
        lmlo, rmlo = tta(x[2], x[3], gen, bg_value)
        return lcc, rcc, lmlo, rmlo

    if "mc_dropout" in methods:
        model.mc(True)
        ps, ls = [], []
        f = feats if not model.backbone_is_stochastic else None
        for _ in range(K_mc):
            o = model.head(f) if f is not None else model(*x)
            ps.append(o["prob"].cpu().numpy()); ls.append(_loc_array(o))
        res["mc_dropout"] = (np.stack(ps), np.stack(ls))
    if "tta" in methods:
        model.mc(False)
        ps, ls = [], []
        for _ in range(K_tta):
            o = model(*tta_inputs())
            ps.append(o["prob"].cpu().numpy()); ls.append(_loc_array(o))
        res["tta"] = (np.stack(ps), np.stack(ls))
    if "mc_tta" in methods:
        model.mc(True)
        ps, ls = [], []
        for _ in range(max(K_mc, K_tta)):
            o = model(*tta_inputs())
            ps.append(o["prob"].cpu().numpy()); ls.append(_loc_array(o))
        res["mc_tta"] = (np.stack(ps), np.stack(ls))
    if "ensemble" in methods and ensemble_heads:
        model.mc(False)
        base = model.head_state_dict()
        ps, ls = [], []
        for h in ensemble_heads:
            model.load_head_state(h)
            o = model.head(feats)
            ps.append(o["prob"].cpu().numpy()); ls.append(_loc_array(o))
        model.load_head_state(base)
        res["ensemble"] = (np.stack(ps), np.stack(ls))
    model.mc(False)
    return res


def run(cfg, checkpoint: str | None = None, split: str | None = None, limit: int | None = None,
        run_name: str | None = None, device=None, model=None) -> Path:
    set_seed(cfg["seed"])
    device = device or get_device()
    run_name = run_name or datetime.now().strftime("uq_%Y%m%d_%H%M%S")
    out_dir = cfg.path("outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exams = exams_wide(pd.read_csv(cfg.path("exam_csv")))
    if split and split != "all":
        exams = exams[exams["split"] == split]
    if limit:
        exams = exams.head(limit)
    if len(exams) == 0:
        raise ValueError("No exams selected - check exam_csv / split")

    model = (model or load_model(cfg, checkpoint)).to(device).eval()
    methods = list(cfg["uq"]["methods"])
    if any(m in methods for m in ("mc_dropout", "mc_tta")) and not model.has_stochastic_layers:
        raise ValueError("MC methods requested but model.head_dropout and backbone_dropout are 0")
    ensemble_heads = load_ensemble_heads(cfg) if "ensemble" in methods else []
    if "ensemble" in methods and len(ensemble_heads) < 2:
        print("NOTE: <2 ensemble members configured - skipping 'ensemble'")
        methods.remove("ensemble")

    pre = Preprocessor.from_config(cfg)
    tta = PairedTTA.from_config(cfg)
    bg_value = (0.0 - pre.img_mean) / pre.img_std  # normalised value of a black background pixel
    info = {"run_name": run_name, "n_exams": int(len(exams)), "methods": methods, "device": str(device),
            "checkpoint": checkpoint, "config": dict(cfg), "conditions": {}, "warnings": []}

    for cond in cfg["uq"]["conditions"]:
        name, sev = cond["name"], int(cond.get("severity", 0))
        tag = name if name == "clean" else f"{name}_s{sev}"
        ds = ExamDataset(exams, pre, None if name == "clean" else name, sev, cfg["seed"])
        dl = DataLoader(ds, batch_size=cfg["uq"]["batch_size"], shuffle=False,
                        num_workers=cfg["uq"]["num_workers"], pin_memory=device.type == "cuda")
        gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        store = {m: ([], []) for m in ["deterministic"] + methods}
        ids, ytc, fu = [], [], []
        t0 = time.time()
        for batch in tqdm(dl, desc=tag):
            r = run_batch(model, batch, methods, cfg, tta, gen, device, ensemble_heads, bg_value)
            for m, (p, l) in r.items():
                store[m][0].append(p); store[m][1].append(l)
            ids += list(batch["exam_id"]); ytc += batch["years_to_cancer"].tolist()
            fu += batch["years_to_last_followup"].tolist()

        arrays = {"exam_id": np.array(ids), "years_to_cancer": np.array(ytc), "years_to_last_followup": np.array(fu)}
        table = exams.set_index(exams["exam_id"].astype(str)).loc[ids].reset_index(drop=True)
        table = table.drop(columns=VIEW_KEYS)
        for m, (ps, ls) in store.items():
            P, L = np.concatenate(ps, 1), np.concatenate(ls, 1)
            arrays[f"{m}_probs"], arrays[f"{m}_locs"] = P, L
            s = summarise_samples(P)
            if m == "deterministic":
                table["p_deterministic"] = P[0]
                continue
            for k, v in s.items():
                table[f"{m}_{k}"] = v
            table[f"{m}_loc_dispersion"] = localisation_dispersion(L)
        np.savez_compressed(out_dir / f"samples_{tag}.npz", **arrays)
        table.to_csv(out_dir / f"predictions_{tag}.csv", index=False)

        p = table["p_deterministic"].to_numpy()
        sanity = {"seconds": round(time.time() - t0, 1), "pred_std": float(p.std()),
                  "pred_min": float(p.min()), "pred_max": float(p.max())}
        if name == "clean" and len(p) > 5 and p.std() < 1e-4:
            msg = (f"Clean predictions are (almost) constant (std={p.std():.2e}). The trained weights "
                   "are probably not loaded or the images are not 16-bit. Results are NOT meaningful.")
            info["warnings"].append(msg); print("WARNING:", msg)
        info["conditions"][tag] = sanity

    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Saved UQ outputs to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="our checkpoint or an original pickled AsymMirai model")
    ap.add_argument("--split", default="all", choices=["all", "calib", "test", "finetune"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.checkpoint, args.split, args.limit, args.run_name)


if __name__ == "__main__":
    main()

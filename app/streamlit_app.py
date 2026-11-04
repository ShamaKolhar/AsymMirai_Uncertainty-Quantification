"""Streamlit demo: per-exam risk, uncertainty, asymmetry heatmap and a 'refer to radiologist' flag.

Two modes:
  * Browse results  - reads an inference run (outputs/<run>/predictions_clean.csv + evaluation/metrics.json),
                      no GPU needed. Good for a portfolio demo.
  * Live inference  - upload 4 views (16-bit PNG) and run the model with MC Dropout + paired TTA
                      (needs the trained weights; slow on CPU).

Run:
    streamlit run app/streamlit_app.py -- --config configs/uq_config.yaml
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from uq.config import load_config  # noqa: E402

DISCLAIMER = ("Research prototype for a dissertation. Not a medical device and not for clinical use. "
              "Uncertainty flags are a statistical aid, not a diagnosis.")


def _args():
    cfg_path = None
    if "--config" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config") + 1]
    return load_config(cfg_path)


def referral_badge(u, cut):
    if cut is None:
        return "—"
    return "🔶 Refer for radiologist review (high uncertainty)" if u > cut else "✅ Confident"


def browse(cfg):
    runs = sorted([p for p in cfg.path("outputs").glob("*") if (p / "predictions_clean.csv").exists()])
    if not runs:
        st.info("No inference runs found in the outputs folder. Run `python -m uq.inference` first.")
        return
    run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name, index=len(runs) - 1)
    df = pd.read_csv(run / "predictions_clean.csv")
    metrics = {}
    mp = run / "evaluation" / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    methods = [m for m in ("mc_tta", "mc_dropout", "tta", "ensemble") if f"{m}_mean" in df]
    method = st.sidebar.selectbox("UQ method", methods)
    cut = metrics.get("methods", {}).get(method, {}).get("refer_cutoff_from_calib")

    if metrics:
        H = metrics["horizon_main"]
        r = metrics["methods"][method]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"AUC ({H}y)", f"{r['auc'][f'{H}y']['auc']:.3f}")
        c2.metric("ECE", f"{r['ece']:.4f}")
        c3.metric("Referred as uncertain", f"{r['test_referral_rate']:.0%}")
        c4.metric("Conformal coverage", f"{r['conformal']['coverage']:.0%}")
        for w in metrics.get("warnings", []):
            st.warning(w)

    split = st.sidebar.selectbox("Split", ["test", "calib", "all"])
    d = df if split == "all" else df[df["split"] == split]
    d = d.assign(uncertainty=d[f"{method}_std"], risk=d[f"{method}_mean"])
    d["flag"] = [referral_badge(u, cut) for u in d["uncertainty"]]
    st.subheader("Exams")
    st.dataframe(d[["exam_id", "risk", "uncertainty", f"{method}_mutual_info", f"{method}_loc_dispersion",
                    "years_to_cancer", "flag"]].sort_values("uncertainty", ascending=False),
                 use_container_width=True, hide_index=True)

    eid = st.selectbox("Inspect exam", d["exam_id"].astype(str))
    sp = run / "samples_clean.npz"
    if sp.exists():
        s = np.load(sp, allow_pickle=True)
        i = int(np.where(s["exam_id"].astype(str) == eid)[0][0])
        st.caption(f"Distribution of the {s[f'{method}_probs'].shape[0]} stochastic risk estimates for this exam")
        st.bar_chart(pd.DataFrame({"risk": s[f"{method}_probs"][:, i]}))
    ev = run / "evaluation"
    figs = [f for f in ("risk_coverage_auc.png", "reliability.png", "uncertainty_hist.png", "conformal_sets.png",
                        "subgroup_referral.png", "robustness_uncertainty.png") if (ev / f).exists()]
    if figs:
        st.subheader("Evaluation figures")
        for f in figs:
            st.image(str(ev / f))


def live(cfg):
    import torch

    from uq.inference import run_batch
    from uq.models.loading import load_model
    from uq.preprocessing import PairedTTA, Preprocessor

    st.write("Upload the four standard screening views as 16-bit PNGs.")
    ups = {k: st.file_uploader(k, type=["png"], key=k) for k in ("L-CC", "R-CC", "L-MLO", "R-MLO")}
    if not all(ups.values()) or not st.button("Run model"):
        return
    try:
        model = load_model(cfg).eval()
    except FileNotFoundError as e:
        st.error(str(e)); return
    pre, tta = Preprocessor.from_config(cfg), PairedTTA.from_config(cfg)
    with tempfile.TemporaryDirectory() as td:
        batch = {}
        for k, f in ups.items():
            p = Path(td) / f"{k}.png"; p.write_bytes(f.getvalue())
            batch[k] = pre(str(p)).unsqueeze(0)
    gen = torch.Generator().manual_seed(cfg["seed"])
    with st.spinner("Running MC Dropout + paired TTA…"):
        res = run_batch(model, batch, ["mc_tta"], cfg, tta, gen, torch.device("cpu"), [], -pre.img_mean / pre.img_std)
    probs = res["mc_tta"][0][:, 0]
    st.metric("Risk (mean)", f"{probs.mean():.3f}")
    st.metric("Uncertainty (std)", f"{probs.std(ddof=1):.4f}")
    with torch.no_grad():
        out = model.mc(False)(*[batch[k] for k in ("L-CC", "R-CC", "L-MLO", "R-MLO")])
    c1, c2 = st.columns(2)
    for col, v in ((c1, "cc"), (c2, "mlo")):
        hm = out[f"{v}_heatmap"][0].numpy()
        col.caption(f"{v.upper()} asymmetry heatmap")
        col.image((hm - hm.min()) / (np.ptp(hm) + 1e-9), clamp=True, use_container_width=True)


def main():
    st.set_page_config(page_title="AsymMirai UQ", layout="wide")
    st.title("Trust-aware AsymMirai: risk + uncertainty")
    st.caption(DISCLAIMER)
    cfg = _args()
    mode = st.sidebar.radio("Mode", ["Browse results", "Live inference"])
    browse(cfg) if mode == "Browse results" else live(cfg)


main()

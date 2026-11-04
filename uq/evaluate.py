"""Evaluate a UQ run: metrics, conformal prediction, robustness, subgroups, figures and a report.

Thresholds (referral cut-off, Youden threshold, conformal quantiles) are ALWAYS fitted on the
`calib` split and reported on the `test` split (patient-disjoint), so nothing is tuned on the
data it is evaluated on.

Usage:
    python -m uq.evaluate --config configs/uq_config.yaml --run outputs/<run_name>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uq import conformal
from uq import metrics as M
from uq.config import load_config
from uq.plotting import (GRID, LABEL_COLOR, METHOD_COLOR, METHOD_LABEL, METHOD_ORDER, NEUTRAL, TEXT2,
                         plt, save, style)

UNC = "std"           # primary uncertainty measure per method (std of the predicted probability)
REFER_FRACTION = 0.10  # refer the 10% most uncertain (cut-off fitted on calib)
SUBGROUP_MIN_N = 30    # groups smaller than this are not reported (estimates too noisy)


def _methods(df):
    return [m for m in METHOD_ORDER if m == "deterministic" or f"{m}_mean" in df]


def _p(df, m):
    return df["p_deterministic"].to_numpy() if m == "deterministic" else df[f"{m}_mean"].to_numpy()


def _u(df, m, measure=UNC):
    return None if m == "deterministic" else df[f"{m}_{measure}"].to_numpy()


def _split(df, cfg, warnings):
    calib, test = df[df["split"] == "calib"], df[df["split"] == "test"]
    if len(calib) == 0 or len(test) == 0:
        warnings.append("calib or test split empty - thresholds fitted and evaluated on the same exams "
                        "(optimistic). Run inference with --split all on a metadata file that has both splits.")
        return df, df
    return calib, test


def _labels(d, h, cfg):
    return M.horizon_labels(d["years_to_cancer"], d["years_to_last_followup"], h,
                            cfg["labels"].get("require_followup_for_negatives", True))


def evaluate_clean(df, cfg, out, warnings):
    calib, test = _split(df, cfg, warnings)
    H = cfg["conformal"]["horizon"]
    methods = _methods(df)
    res = {"n_calib": len(calib), "n_test": len(test), "horizon_main": H, "methods": {}}
    yc, mc = _labels(calib, H, cfg)
    yt, mt = _labels(test, H, cfg)
    res["n_test_used_main_horizon"] = int(mt.sum())
    res["n_test_cancers_main_horizon"] = int(yt[mt].sum())
    rc_tables, sub_tables = {}, {}
    for m in methods:
        pt, pc = _p(test, m), _p(calib, m)
        r = {"auc": {}, "c_index": M.c_index(test["years_to_cancer"], test["years_to_last_followup"], pt,
                                             cfg["labels"]["no_cancer_value"])}
        for h in cfg["labels"]["horizons"]:
            y, mask = _labels(test, h, cfg)
            a, lo, hi = M.auc_with_ci(y[mask], pt[mask], seed=cfg["seed"])
            r["auc"][f"{h}y"] = {"auc": a, "ci_low": lo, "ci_high": hi, "n": int(mask.sum()), "cancers": int(y[mask].sum())}
        r["ece"] = M.ece(yt[mt], pt[mt]); r["brier"] = M.brier(yt[mt], pt[mt])
        cm = conformal.fit(pc[mc], yc[mc], cfg["conformal"]["alpha"], mondrian=True)
        r["conformal"] = {**conformal.evaluate(pt[mt], yt[mt], cm), "q0": cm["q0"], "q1": cm["q1"]}
        if m != "deterministic":
            ut, uc = _u(test, m), _u(calib, m)
            rc = M.risk_coverage(yt[mt], pt[mt], ut[mt]); rc_tables[m] = rc
            thr = M.youden_threshold(yc[mc], pc[mc]) if 0 < yc[mc].sum() < mc.sum() else 0.5
            refer_cut = float(np.quantile(uc, 1 - REFER_FRACTION))
            r["risk_coverage"] = rc.to_dict("records")
            r["aurc_brier"] = M.aurc(rc, "brier")
            r["uncertainty_error"] = M.uncertainty_error(yt[mt], pt[mt], ut[mt], thr)
            r["refer_cutoff_from_calib"] = refer_cut
            r["test_referral_rate"] = float((ut > refer_cut).mean())
            for meas in ("std", "mutual_info", "entropy"):
                if f"{m}_{meas}" in test:
                    r[f"mean_{meas}"] = float(test[f"{m}_{meas}"].mean())
            r["uncertainty_by_label"] = {"cancer": float(ut[mt][yt[mt] == 1].mean()) if yt[mt].any() else None,
                                         "no_cancer": float(ut[mt][yt[mt] == 0].mean())}
            ld = test[f"{m}_loc_dispersion"].to_numpy()
            r["localisation"] = {"mean_dispersion": float(ld.mean()),
                                 "spearman_with_uncertainty": float(spearmanr(ld, ut).statistic) if ld.std() > 0 and ut.std() > 0 else None}
            st = M.subgroup_table(test[mt], yt[mt], pt[mt], ut[mt], refer_cut, SUBGROUP_MIN_N); sub_tables[m] = st
            st.to_csv(out / f"subgroups_{m}.csv", index=False)
        res["methods"][m] = r
    return res, calib, test, rc_tables, sub_tables


def evaluate_robustness(run_dir, cfg, methods):
    rows = []
    for f in sorted(run_dir.glob("predictions_*.csv")):
        tag = f.stem.replace("predictions_", "")
        mt = re.match(r"(.+)_s(\d+)$", tag)
        corr, sev = (mt.group(1), int(mt.group(2))) if mt else ("clean", 0)
        d = pd.read_csv(f)
        d = d[d["split"] == "test"] if (d["split"] == "test").any() else d
        y, mask = _labels(d, cfg["conformal"]["horizon"], cfg)
        for m in methods:
            if m != "deterministic" and f"{m}_mean" not in d:
                continue
            p = _p(d, m)
            rows.append({"condition": tag, "corruption": corr, "severity": sev, "method": m,
                         "auc": M._safe_auc(y[mask], p[mask]), "brier": M.brier(y[mask], p[mask]),
                         "mean_uncertainty": float(d[f"{m}_{UNC}"].mean()) if m != "deterministic" else np.nan,
                         "mean_loc_dispersion": float(d[f"{m}_loc_dispersion"].mean()) if m != "deterministic" else np.nan})
    rob = pd.DataFrame(rows)
    trend = {}
    for corr in rob["corruption"].unique():
        if corr == "clean":
            continue
        for m in rob["method"].unique():
            if m == "deterministic":
                continue
            s = rob[(rob["method"] == m) & rob["corruption"].isin(["clean", corr])].sort_values("severity")
            if len(s) >= 3:
                trend[f"{m}|{corr}"] = float(spearmanr(s["severity"], s["mean_uncertainty"]).statistic)
    return rob, trend


# ----------------------------------------------------------------------------- figures
def figures(out, cfg, res, calib, test, rc_tables, sub_tables, rob):
    style()
    from sklearn.metrics import roc_curve

    H = cfg["conformal"]["horizon"]
    yt, mt = _labels(test, H, cfg)
    methods = list(res["methods"].keys())
    unc_methods = [m for m in methods if m != "deterministic"]
    primary = "mc_tta" if "mc_tta" in unc_methods else (unc_methods[0] if unc_methods else None)
    paths = []

    # 1. ROC
    if 0 < yt[mt].sum() < mt.sum():
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
        for m in methods:
            fpr, tpr, _ = roc_curve(yt[mt], _p(test, m)[mt])
            ax.plot(fpr, tpr, color=METHOD_COLOR[m], label=f"{METHOD_LABEL[m]} (AUC {res['methods'][m]['auc'][f'{H}y']['auc']:.3f})")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC, cancer within {H} years (test)")
        ax.legend(loc="lower right"); paths.append(save(fig, out / "roc.png"))

    # 2. Reliability
    fig, ax = plt.subplots(figsize=(5.5, 5))
    mx = 0
    for m in methods:
        r = M.reliability(yt[mt], _p(test, m)[mt])
        ax.plot(r["p_mean"], r["y_rate"], marker="o", color=METHOD_COLOR[m], label=f"{METHOD_LABEL[m]} (ECE {res['methods'][m]['ece']:.3f})")
        mx = max(mx, r["p_mean"].max(), r["y_rate"].max())
    ax.plot([0, mx], [0, mx], color=NEUTRAL, lw=1, ls="--", label="Perfect calibration")
    ax.set(xlabel="Mean predicted risk (bin)", ylabel="Observed cancer rate", title=f"Calibration, {H}-year risk (test)")
    ax.legend(); paths.append(save(fig, out / "reliability.png"))

    # 3. Risk-coverage (AUC) and (Brier) - two charts, never a dual axis
    for col, ttl, fn in (("auc", "AUC of retained exams", "risk_coverage_auc.png"),
                         ("brier", "Brier score of retained exams (lower is better)", "risk_coverage_brier.png")):
        if not rc_tables:
            break
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for m, rc in rc_tables.items():
            ax.plot(rc["coverage"], rc[col], marker="o", color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        full = res["methods"]["deterministic"]["auc"][f"{H}y"]["auc"] if col == "auc" else res["methods"]["deterministic"]["brier"]
        ax.axhline(full, color=NEUTRAL, ls="--", lw=1, label="All exams, deterministic")
        ax.set(xlabel="Coverage (fraction of exams kept; most uncertain referred)", ylabel=col.upper() if col == "auc" else "Brier",
               title=ttl); ax.invert_xaxis(); ax.legend(); paths.append(save(fig, out / fn))

    if primary:
        u = _u(test, primary)
        # 4. Uncertainty distribution by label
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        pos = u[mt][yt[mt] == 1]; neg = u[mt][yt[mt] == 0]
        bins = np.histogram_bin_edges(u[mt], bins=40)
        ax.hist(neg, bins=bins, density=True, histtype="step", lw=2, color=LABEL_COLOR[0], label=f"No cancer (n={len(neg)})")
        if len(pos):
            ax.hist(pos, bins=bins, density=True, histtype="step", lw=2, color=LABEL_COLOR[1], label=f"Cancer within {H}y (n={len(pos)})")
        ax.axvline(res["methods"][primary]["refer_cutoff_from_calib"], color=TEXT2, ls="--", lw=1, label="Referral cut-off (calib, top 10%)")
        ax.set(xlabel=f"{METHOD_LABEL[primary]} uncertainty (std of predicted risk)", ylabel="Density",
               title="Uncertainty by outcome (test)"); ax.legend(); paths.append(save(fig, out / "uncertainty_hist.png"))

        # 5. Predicted risk vs uncertainty
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        p = _p(test, primary)
        for lab in (0, 1):
            s = (yt == lab) & mt
            if s.any():
                ax.scatter(p[s], u[s], s=14 if lab == 0 else 28, alpha=0.5 if lab == 0 else 0.9, color=LABEL_COLOR[lab],
                           edgecolors=GRID if lab else "none", linewidths=0.5,
                           label="No cancer" if lab == 0 else f"Cancer within {H}y")
        ax.axhline(res["methods"][primary]["refer_cutoff_from_calib"], color=TEXT2, ls="--", lw=1, label="Referral cut-off")
        ax.set(xlabel="Mean predicted risk", ylabel="Uncertainty (std)", title=f"Predicted risk vs uncertainty - {METHOD_LABEL[primary]}")
        ax.legend(); paths.append(save(fig, out / "confidence_vs_uncertainty.png"))

        # 6. Localisation dispersion
        ld = test[f"{primary}_loc_dispersion"].to_numpy()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bins = np.linspace(0, max(ld.max(), 1e-3), 21)
        ax.hist(ld[mt & (yt == 0)], bins=bins, density=True, histtype="step", lw=2, color=LABEL_COLOR[0], label="No cancer")
        if (mt & (yt == 1)).any():
            ax.hist(ld[mt & (yt == 1)], bins=bins, density=True, histtype="step", lw=2, color=LABEL_COLOR[1], label=f"Cancer within {H}y")
        ax.set(xlabel="Fraction of samples whose most-asymmetric window differs from the modal one",
               ylabel="Density", title="Localisation uncertainty (test)"); ax.legend(); paths.append(save(fig, out / "localisation_uncertainty.png"))

        # 7. Subgroup referral rates
        st = sub_tables.get(primary)
        if st is not None and len(st):
            st = st.iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(st) + 1.2))
            labels = [f"{a.replace('_band', '')}: {g} (n={n})" for a, g, n in zip(st["attribute"], st["group"], st["n"])]
            vals = st["referral_rate"].to_numpy() * 100
            ax.barh(labels, vals, color=METHOD_COLOR[primary], height=0.6, edgecolor="#fcfcfb", linewidth=2)
            for yi, v in enumerate(vals):
                ax.text(v + 0.3, yi, f"{v:.0f}%", va="center", fontsize=8, color=TEXT2)
            ax.axvline(REFER_FRACTION * 100, color=TEXT2, ls="--", lw=1, label="Overall target (10%)")
            ax.set_xlim(0, max(vals.max() * 1.15, REFER_FRACTION * 100 * 1.5))
            ax.set(xlabel="Exams referred as uncertain (%)", title="Referral rate by subgroup (test)")
            ax.grid(axis="y", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(st)))
            paths.append(save(fig, out / "subgroup_referral.png"))

    # 8. Conformal sets
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = [METHOD_LABEL[m] for m in methods][::-1]
    keys = [("frac_set_{0}", "{no cancer}"), ("frac_set_{0,1}_refer", "{both} = refer"), ("frac_set_{1}", "{cancer}")]
    left = np.zeros(len(methods))
    for i, (k, lab) in enumerate(keys):
        vals = np.array([res["methods"][m]["conformal"][k] for m in methods][::-1]) * 100
        ax.barh(names, vals, left=left, color=[LABEL_COLOR[0], NEUTRAL, LABEL_COLOR[1]][i], height=0.6,
                edgecolor="#fcfcfb", linewidth=2, label=lab)
        left += vals
    ax.set(xlabel="% of test exams", title=f"Conformal prediction sets ({int(100 * (1 - cfg['conformal']['alpha']))}% target coverage)")
    ax.grid(axis="y", visible=False); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    paths.append(save(fig, out / "conformal_sets.png"))

    # 9. Robustness: uncertainty vs severity, and AUC vs severity (separate charts)
    if rob is not None and len(rob) and (rob["corruption"] != "clean").any():
        for col, ttl, fn in (("mean_uncertainty", "Mean uncertainty vs corruption severity", "robustness_uncertainty.png"),
                             ("auc", "AUC vs corruption severity", "robustness_auc.png")):
            corrs = [c for c in rob["corruption"].unique() if c != "clean"]
            fig, axes = plt.subplots(1, len(corrs), figsize=(max(4.2 * len(corrs), 7), 4.2), sharey=True, squeeze=False)
            for ax, corr in zip(axes[0], corrs):
                for m in [m for m in METHOD_ORDER if m in rob["method"].unique()]:
                    if col == "mean_uncertainty" and m == "deterministic":
                        continue
                    s = rob[(rob["method"] == m) & rob["corruption"].isin(["clean", corr])].sort_values("severity")
                    ax.plot(s["severity"], s[col], marker="o", color=METHOD_COLOR[m], label=METHOD_LABEL[m])
                ax.set(title=corr.replace("_", " ").capitalize(), xlabel="Severity (0 = clean)")
                ax.set_xticks(sorted(rob[rob["corruption"].isin(["clean", corr])]["severity"].unique()))
            axes[0][0].set_ylabel("Mean std of predicted risk" if col == "mean_uncertainty" else "AUC")
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), bbox_to_anchor=(0.5, -0.02))
            fig.suptitle(ttl, x=0.02, ha="left", fontweight="bold", color="#0b0b0b")
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))
            paths.append(save(fig, out / fn))
    return paths


def write_report(out, res, trend, warnings):
    H = res["horizon_main"]
    L = [f"# UQ evaluation report", "", f"Calibration exams: {res['n_calib']} · Test exams: {res['n_test']} "
         f"(used at {H}y: {res['n_test_used_main_horizon']}, cancers: {res['n_test_cancers_main_horizon']})", ""]
    for w in warnings:
        L.append(f"> **Warning:** {w}")
    L += ["", "## Discrimination and calibration (test)", "",
          "| Method | " + " | ".join(f"AUC {h}" for h in next(iter(res['methods'].values()))["auc"]) + " | C-index | ECE | Brier |",
          "|---|" + "---|" * (len(next(iter(res['methods'].values()))["auc"]) + 3)]
    for m, r in res["methods"].items():
        aucs = " | ".join(f"{v['auc']:.3f} [{v['ci_low']:.3f}-{v['ci_high']:.3f}]" if np.isfinite(v["auc"]) else "n/a"
                          for v in r["auc"].values())
        L.append(f"| {METHOD_LABEL[m]} | {aucs} | {r['c_index']:.3f} | {r['ece']:.4f} | {r['brier']:.4f} |")
    L += ["", "## Is the uncertainty useful? (test)", "",
          "| Method | AURC (Brier, ↓) | AUROC unc→error | Spearman unc~sq.err | Unc. cancer / no cancer | Referral rate | Loc. dispersion |",
          "|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        if m == "deterministic":
            continue
        ue, ub = r["uncertainty_error"], r["uncertainty_by_label"]
        L.append(f"| {METHOD_LABEL[m]} | {r['aurc_brier']:.5f} | {ue['auroc_uncertainty_vs_error']:.3f} | "
                 f"{ue['spearman_uncertainty_sqerror']:.3f} | {ub['cancer'] if ub['cancer'] is None else round(ub['cancer'], 5)} / "
                 f"{ub['no_cancer']:.5f} | {r['test_referral_rate']:.1%} | {r['localisation']['mean_dispersion']:.3f} |")
    L += ["", f"## Conformal prediction ({H}y, class-conditional)", "",
          "| Method | Coverage | Coverage (cancer) | {no cancer} | {cancer} | {both} = refer |", "|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        c = r["conformal"]
        L.append(f"| {METHOD_LABEL[m]} | {c['coverage']:.3f} | {c['coverage_cancer']:.3f} | {c['frac_set_{0}']:.1%} | "
                 f"{c['frac_set_{1}']:.1%} | {c['frac_set_{0,1}_refer']:.1%} |")
    if trend:
        L += ["", "## Robustness: does uncertainty rise with corruption severity?", "",
              "Spearman correlation between severity and mean uncertainty (1.0 = rises monotonically).", ""]
        L += [f"- {METHOD_LABEL[k.split('|')[0]]} / {k.split('|')[1]}: {v:.2f}" for k, v in trend.items()]
    L += ["", "Figures: roc.png, reliability.png, risk_coverage_auc.png, risk_coverage_brier.png, uncertainty_hist.png, "
          "confidence_vs_uncertainty.png, localisation_uncertainty.png, subgroup_referral.png, conformal_sets.png, "
          "robustness_uncertainty.png, robustness_auc.png"]
    (out / "report.md").write_text("\n".join(L))


def run(cfg, run_dir) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "evaluation"
    out.mkdir(exist_ok=True)
    warnings = []
    info_p = run_dir / "run_info.json"
    if info_p.exists():
        warnings += json.loads(info_p.read_text()).get("warnings", [])
    df = pd.read_csv(run_dir / "predictions_clean.csv")
    res, calib, test, rc_tables, sub_tables = evaluate_clean(df, cfg, out, warnings)
    rob, trend = evaluate_robustness(run_dir, cfg, list(res["methods"].keys()))
    rob.to_csv(out / "robustness.csv", index=False)
    res["robustness_trend_spearman"] = trend
    res["warnings"] = warnings
    figures(out, cfg, res, calib, test, rc_tables, sub_tables, rob)
    with open(out / "metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    write_report(out, res, trend, warnings)
    print(f"Evaluation written to {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True, help="outputs/<run_name> directory from uq.inference")
    args = ap.parse_args()
    run(load_config(args.config), args.run)


if __name__ == "__main__":
    main()

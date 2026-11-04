"""Risk-prediction and uncertainty metrics.

Replaces the old "95th-percentile of variance = ambiguous" rule (which flags exactly 5% of
cases by construction, whatever the model does) with metrics that test whether the
uncertainty is *useful*:
  * discrimination: AUC at 1-5 year horizons (bootstrap 95% CI), Harrell's C-index
  * calibration: ECE (equal-frequency bins - prevalence is low), Brier score, reliability curve
  * selective prediction: risk-coverage curves (does AUC/Brier improve as uncertain cases are referred?)
  * uncertainty-error: AUROC of uncertainty for detecting misclassifications, Spearman(uncertainty, squared error)
  * subgroup breakdowns (race, breast density, age)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------------------------------------------------------- labels
def horizon_labels(years_to_cancer, followup, horizon: float, require_followup: bool = True):
    """Returns (y, mask): y=1 cancer within horizon; mask=False for censored negatives."""
    ytc = np.asarray(years_to_cancer, float)
    fu = np.asarray(followup, float)
    y = (ytc <= horizon).astype(int)
    if require_followup:
        mask = (y == 1) | (np.nan_to_num(fu, nan=-1) >= horizon)
    else:
        mask = np.ones_like(y, bool)
    return y, mask


def _safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def auc_with_ci(y, p, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    y, p = np.asarray(y), np.asarray(p)
    point = _safe_auc(y, p)
    if np.isnan(point):
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = []
    for _ in range(n_boot):  # stratified bootstrap keeps both classes present
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def c_index(years_to_cancer, followup, risk, no_cancer_value: float = 100):
    from lifelines.utils import concordance_index

    ytc, fu = np.asarray(years_to_cancer, float), np.asarray(followup, float)
    event = ytc < no_cancer_value
    t = np.where(event, ytc, fu)
    ok = np.isfinite(t) & (t >= 0)
    if event[ok].sum() == 0:
        return np.nan
    return float(concordance_index(t[ok], -np.asarray(risk)[ok], event[ok]))


# ----------------------------------------------------------------------------- calibration
def reliability(y, p, n_bins: int = 10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    rows = []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "p_mean": float(p[m].mean()), "y_rate": float(y[m].mean())})
    return pd.DataFrame(rows)


def ece(y, p, n_bins: int = 10):
    r = reliability(y, p, n_bins)
    return float((r["n"] * (r["p_mean"] - r["y_rate"]).abs()).sum() / r["n"].sum()) if len(r) else np.nan


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# ----------------------------------------------------------------------------- uncertainty usefulness
def risk_coverage(y, p, u, coverages=np.linspace(0.5, 1.0, 11)):
    """Keep the `coverage` fraction of LEAST uncertain exams; refer the rest. Lower u = more certain."""
    y, p, u = map(np.asarray, (y, p, u))
    order = np.argsort(u, kind="stable")
    rows = []
    for c in coverages:
        k = max(int(round(c * len(y))), 1)
        keep = order[:k]
        rows.append({"coverage": float(c), "n": k, "auc": _safe_auc(y[keep], p[keep]),
                     "brier": brier(y[keep], p[keep]), "cancers_kept": int(y[keep].sum())})
    return pd.DataFrame(rows)


def aurc(rc: pd.DataFrame, col: str = "brier") -> float:
    """Area under the risk(=Brier)-coverage curve; lower is better."""
    x, yv = rc["coverage"].to_numpy(), rc[col].to_numpy()
    return float(np.sum((x[1:] - x[:-1]) * (yv[1:] + yv[:-1]) / 2))


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def uncertainty_error(y, p, u, threshold: float):
    y, p, u = map(np.asarray, (y, p, u))
    wrong = ((p >= threshold).astype(int) != y).astype(int)
    sq = (p - y) ** 2
    rho = spearmanr(u, sq).statistic if np.std(u) > 0 else np.nan
    return {"misclassification_rate": float(wrong.mean()),
            "auroc_uncertainty_vs_error": _safe_auc(wrong, u),
            "spearman_uncertainty_sqerror": float(rho)}


# ----------------------------------------------------------------------------- subgroups
def density_label(v):
    return {1.0: "A (fatty)", 2.0: "B (scattered)", 3.0: "C (heterogeneous)", 4.0: "D (dense)"}.get(v, "unknown")


def age_band(a):
    if pd.isna(a):
        return "unknown"
    return "<50" if a < 50 else ("50-64" if a < 65 else "65+")


def subgroup_table(df: pd.DataFrame, y, p, u, refer_threshold: float, min_n: int = 30) -> pd.DataFrame:
    d = df.copy()
    d["_y"], d["_p"], d["_u"] = y, p, u
    d["density"] = d.get("tissueden", pd.Series(np.nan, index=d.index)).map(density_label)
    d["age_band"] = d.get("age_at_study", pd.Series(np.nan, index=d.index)).map(age_band)
    rows = []
    for col in ("race", "density", "age_band"):
        if col not in d:
            continue
        for g, s in d.groupby(col, dropna=False):
            if len(s) < min_n:
                continue
            rows.append({"attribute": col, "group": str(g), "n": len(s), "cancers": int(s["_y"].sum()),
                         "auc": _safe_auc(s["_y"], s["_p"]), "ece": ece(s["_y"], s["_p"]),
                         "mean_uncertainty": float(s["_u"].mean()),
                         "referral_rate": float((s["_u"] > refer_threshold).mean())})
    return pd.DataFrame(rows)

"""Choose a small, *enriched* subset of exams that is still evaluable on a free GPU.

At ~1-2% cancer prevalence, a random 400 exams would contain only a handful of cancers, too
few for an AUC or an uncertainty comparison. This builds a nested case-control sample:

  * cases    = exams with cancer diagnosed within `subset.horizon` years (one exam per patient,
               the one closest to diagnosis)
  * controls = exams from patients never diagnosed with cancer (one exam per patient),
               `controls_per_case` per case, drawn from the SAME split (calib/test) and, where
               possible, the same age band as the case
  * capped at `subset.max_exams` in total

Caveat (written into the CSV and shown in the evaluation report): enrichment changes the
prevalence. Ranking metrics (AUC, C-index, risk-coverage, uncertainty-vs-error) remain valid;
calibration (ECE/Brier) and conformal set sizes reflect the enriched sample, not a screening
population.

Usage:
    python -m uq.data.select_subset --config configs/free_tier.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from uq.config import load_config
from uq.metrics import age_band


def select(df_long: pd.DataFrame, horizon: float = 5, controls_per_case: int = 3, max_exams: int = 400,
           max_cases: int | None = None, no_cancer_value: float = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ex = df_long.drop_duplicates("exam_id").copy()
    ex["_age_band"] = ex["age_at_study"].map(age_band) if "age_at_study" in ex else "unknown"

    cases = ex[ex["years_to_cancer"] <= horizon]
    cases = cases.sort_values("years_to_cancer").drop_duplicates("patient_id", keep="first")
    pool = ex[ex["years_to_cancer"] >= no_cancer_value]
    pool = pool.sample(frac=1.0, random_state=seed).drop_duplicates("patient_id", keep="first")

    n_cases = min(len(cases), max_exams // (1 + controls_per_case))
    if max_cases:
        n_cases = min(n_cases, max_cases)
    if n_cases == 0:
        raise ValueError(f"No cancer cases within {horizon} years in this table - nothing to enrich")
    if n_cases < len(cases):  # stratify by split so calib (thresholds, conformal) AND test both get cases
        share = cases["split"].value_counts() / len(cases)
        quota = (share * n_cases).round().astype(int).clip(lower=1)
        cases = pd.concat([g.sample(n=min(len(g), int(quota[sp])), random_state=seed)
                           for sp, g in cases.groupby("split")])

    chosen, used = [], set()
    for _, c in cases.iterrows():
        cand = pool[(pool["split"] == c["split"]) & ~pool["exam_id"].isin(used)]
        same_age = cand[cand["_age_band"] == c["_age_band"]]
        src = same_age if len(same_age) >= controls_per_case else cand
        k = min(controls_per_case, len(src))
        if k:
            pick = src.iloc[rng.choice(len(src), size=k, replace=False)]
            chosen.append(pick)
            used.update(pick["exam_id"])
    controls = pd.concat(chosen) if chosen else pool.iloc[:0]

    keep = set(cases["exam_id"]) | set(controls["exam_id"])
    out = df_long[df_long["exam_id"].isin(keep)].copy()
    out["enriched"] = True
    return out


def run(cfg) -> pd.DataFrame:
    s = cfg["subset"]
    src = cfg.path("planned_exam_csv")
    df = pd.read_csv(src)
    out = select(df, s.get("horizon", 5), s.get("controls_per_case", 3), s.get("max_exams", 400),
                 s.get("max_cases"), cfg["labels"]["no_cancer_value"], s.get("seed", cfg["seed"]))
    dst = cfg.path("exam_csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    ex = out.drop_duplicates("exam_id")
    h = s.get("horizon", 5)
    info = {"source": str(src), "exams": int(len(ex)), "images": int(len(out)),
            "cases_within_horizon": int((ex["years_to_cancer"] <= h).sum()),
            "controls": int((ex["years_to_cancer"] >= cfg["labels"]["no_cancer_value"]).sum()),
            "by_split": ex["split"].value_counts().to_dict(), "horizon": h}
    dst.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"Wrote {dst}: {info['exams']} exams ({info['cases_within_horizon']} cases, "
          f"{info['controls']} controls), {info['images']} images to download. Splits: {info['by_split']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

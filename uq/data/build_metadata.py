"""Build the exam-level CSV AsymMirai expects (same columns as AsymMirai/example_data_format.csv).

Fixes vs. the original notebook:
  * Exams are grouped by accession number (`acc_anon`), not DICOM PatientID. PatientID is
    the *patient*, so several exams from different years were merged and a "4-view exam"
    could mix an L-CC from 2014 with an R-CC from 2019.
  * Images are joined to EMBED metadata with a pandas merge on `anon_dicom_path`
    (the old code ran `str.contains` per PNG, O(n^2), and could match the wrong file).
  * Only standard 2D screening views are kept: FinalImageType == '2D' (no synthetic
    C-view), no spot-compression/magnification views, ViewPosition in {CC, MLO}.
  * Labels: `years_to_cancer` (100 = no cancer, AsymMirai convention) and
    `years_to_last_followup`, derived from the clinical table
    (path_severity 0 = invasive, 1 = non-invasive cancer). Exams on/after the
    diagnosis date are dropped.
  * Subgroup columns (age, race, breast density) are carried along for fairness analysis.
  * A patient-level split column (`split` = calib / test / finetune) so no patient leaks
    between the calibration set (thresholds, conformal) and the evaluation set.

Usage:
    python -m uq.data.build_metadata --config configs/uq_config.yaml
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from uq.config import load_config

CANCER_SEVERITIES = {0.0, 1.0}  # EMBED: 0 invasive cancer, 1 non-invasive cancer (DCIS)
VIEWS = ("CC", "MLO")
SIDES = ("L", "R")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed").dt.normalize()


def patient_split(patient_ids: pd.Series, calib_fraction: float, finetune_fraction: float, seed: int) -> pd.Series:
    """Deterministic patient-level split via hashing (stable across runs and machines)."""

    def bucket(pid) -> float:
        h = hashlib.sha256(f"{seed}-{pid}".encode()).hexdigest()
        return int(h[:12], 16) / float(16 ** 12)

    u = patient_ids.map(bucket)
    out = np.where(u < finetune_fraction, "finetune",
                   np.where(u < finetune_fraction + calib_fraction, "calib", "test"))
    return pd.Series(out, index=patient_ids.index)


def derive_exam_labels(clinical: pd.DataFrame, no_cancer_value: float = 100) -> pd.DataFrame:
    """One row per exam (acc_anon) with years_to_cancer, years_to_last_followup and subgroup columns."""
    c = clinical.copy()
    c["study_date"] = _to_date(c["study_date_anon"])
    c["proc_date"] = _to_date(c["procdate_anon"]) if "procdate_anon" in c else pd.NaT

    cancer = c[c["path_severity"].isin(CANCER_SEVERITIES)].copy()
    cancer["dx_date"] = cancer["proc_date"].fillna(cancer["study_date"])
    first_dx = cancer.groupby("empi_anon")["dx_date"].min().rename("first_cancer_date")
    last_seen = c.groupby("empi_anon")["study_date"].max().rename("last_study_date")

    agg = {"empi_anon": "first", "study_date": "first", "age_at_study": "first",
           "race": "first", "tissueden": "first", "desc": "first"}
    if "screen_exam" in c:
        agg["screen_exam"] = "max"
    exams = c.groupby("acc_anon").agg(agg).reset_index()
    exams = exams.join(first_dx, on="empi_anon").join(last_seen, on="empi_anon")

    days_to_cancer = (exams["first_cancer_date"] - exams["study_date"]).dt.days
    exams["years_to_cancer"] = np.where(exams["first_cancer_date"].isna(), no_cancer_value, days_to_cancer / 365.25)
    exams["years_to_last_followup"] = (exams["last_study_date"] - exams["study_date"]).dt.days / 365.25
    # Exams on/after the diagnosis are not risk-prediction exams
    post_dx = exams["first_cancer_date"].notna() & (days_to_cancer <= 0)
    exams = exams[~post_dx].copy()
    return exams


def build(cfg) -> pd.DataFrame:
    index = pd.read_csv(cfg.path("image_index_csv"))
    index = index[index["error"].isna() & index["png_path"].notna()]

    meta_cols = ["anon_dicom_path", "empi_anon", "acc_anon", "study_date_anon",
                 "ImageLateralityFinal", "ViewPosition", "FinalImageType", "spot_mag"]
    meta = pd.read_csv(cfg.path("metadata_csv"), usecols=lambda col: col in meta_cols, low_memory=False)

    imgs = index.merge(meta, on="anon_dicom_path", how="inner", validate="many_to_one")
    n0 = len(imgs)
    imgs = imgs[(imgs["FinalImageType"] == "2D")
                & (imgs["spot_mag"].isna())
                & (imgs["ViewPosition"].isin(VIEWS))
                & (imgs["ImageLateralityFinal"].isin(SIDES))]
    print(f"Images matched to metadata: {n0}; standard 2D CC/MLO views kept: {len(imgs)}")

    imgs = imgs.rename(columns={"acc_anon": "exam_id", "empi_anon": "patient_id",
                                "ImageLateralityFinal": "laterality", "ViewPosition": "view",
                                "png_path": "file_path"})
    # one image per exam/side/view, deterministic
    imgs = imgs.sort_values("file_path").drop_duplicates(["exam_id", "laterality", "view"], keep="last")
    complete = imgs.groupby("exam_id")["file_path"].transform("count") == 4
    imgs = imgs[complete]
    print(f"Complete 4-view exams: {imgs['exam_id'].nunique()} from {imgs['patient_id'].nunique()} patients")

    labels = derive_exam_labels(pd.read_csv(cfg.path("clinical_csv"), low_memory=False),
                                cfg["labels"]["no_cancer_value"])
    if cfg["labels"].get("screening_only", True) and "screen_exam" in labels:
        labels = labels[labels["screen_exam"].astype(str).str.lower() == "true"]
    labels = labels.rename(columns={"acc_anon": "exam_id"}).drop(columns=["empi_anon"])

    df = imgs.merge(labels, on="exam_id", how="inner")
    df["spot_mag"] = np.nan
    df["matched_image"] = np.nan
    df["split"] = patient_split(df["patient_id"], cfg["split"]["calib_fraction"],
                                cfg["split"]["finetune_fraction"], cfg["seed"])
    cols = ["exam_id", "patient_id", "laterality", "view", "file_path", "years_to_cancer",
            "years_to_last_followup", "desc", "spot_mag", "matched_image", "split",
            "study_date", "age_at_study", "race", "tissueden"]
    df = df[cols].sort_values(["patient_id", "exam_id", "laterality", "view"]).reset_index(drop=True)

    out = cfg.path("exam_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ex = df.drop_duplicates("exam_id")
    pos5 = (ex["years_to_cancer"] <= 5).sum()
    print(f"Wrote {out}: {len(ex)} exams, {ex['patient_id'].nunique()} patients, "
          f"{pos5} exams with cancer within 5 years")
    print(ex["split"].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()

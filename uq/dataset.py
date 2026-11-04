"""Exam-level dataset (one item = one 4-view screening exam)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from uq.preprocessing import Preprocessor

VIEW_KEYS = ["L-CC", "R-CC", "L-MLO", "R-MLO"]
EXAM_COLS = ["exam_id", "patient_id", "years_to_cancer", "years_to_last_followup", "split",
             "age_at_study", "race", "tissueden", "study_date", "enriched"]


def exams_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Long (one row per image) -> wide (one row per exam with L-CC/R-CC/L-MLO/R-MLO path columns)."""
    d = df_long.copy()
    d["side_view"] = d["laterality"] + "-" + d["view"]
    d = d.sort_values("file_path").drop_duplicates(["exam_id", "side_view"], keep="last")
    wide = d.pivot(index="exam_id", columns="side_view", values="file_path")
    for k in VIEW_KEYS:
        if k not in wide:
            wide[k] = np.nan
    meta = d[[c for c in EXAM_COLS if c in d.columns]].drop_duplicates("exam_id").set_index("exam_id")
    return wide[VIEW_KEYS].join(meta).reset_index()


def _seed_for(exam_id, tag: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{exam_id}|{tag}".encode()).hexdigest()
    return int(h[:8], 16)


class ExamDataset(Dataset):
    def __init__(self, exams: pd.DataFrame, preprocessor: Preprocessor,
                 corruption: str | None = None, severity: int = 0, seed: int = 42):
        self.exams = exams.reset_index(drop=True)
        self.pre = preprocessor
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.exams)

    def __getitem__(self, i):
        row = self.exams.iloc[i]
        tag = f"{self.corruption}-{self.severity}"
        imgs = {}
        for k in VIEW_KEYS:
            rng = np.random.default_rng(_seed_for(row["exam_id"], f"{tag}-{k}", self.seed))
            imgs[k] = self.pre(row[k], self.corruption, self.severity, rng)
        ytc = float(row.get("years_to_cancer", 100.0))
        fu = float(row.get("years_to_last_followup", np.nan))
        return {
            "exam_id": str(row["exam_id"]),
            "years_to_cancer": torch.tensor(ytc),
            "years_to_last_followup": torch.tensor(fu),
            **{k: v for k, v in imgs.items()},
        }

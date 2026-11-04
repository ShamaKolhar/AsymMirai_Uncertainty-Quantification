"""End-to-end: DICOM -> 16-bit PNG -> exam CSV -> fine-tune heads -> UQ inference -> evaluation."""
import json

import cv2
import numpy as np
import pandas as pd
import torch

from conftest import make_original
from synthetic import make
from uq.config import load_config


def _cfg(root):
    return load_config(overrides={
        "root": str(root),
        "paths": {"weights": {"asymmirai_full_model": "weights/full.pt", "ensemble_members": []}},
        "preprocessing": {"target_size": [320, 256]},
        "split": {"calib_fraction": 0.35, "finetune_fraction": 0.25},
        "labels": {"require_followup_for_negatives": False, "horizons": [1, 2, 3]},
        "conformal": {"horizon": 3, "alpha": 0.2},
        "model": {"head_dropout": 0.3},
        "uq": {"mc_samples": 3, "tta_samples": 3, "batch_size": 4, "num_workers": 0,
               "methods": ["mc_dropout", "tta", "mc_tta", "ensemble"],
               "conditions": [{"name": "clean"}, {"name": "gaussian", "severity": 1},
                              {"name": "gaussian", "severity": 3}]},
    })


def test_end_to_end(no_cuda, tmp_path):
    from uq.data import build_metadata, dicom_to_png
    from uq import evaluate, finetune_head, inference
    from uq.config import get_device

    make(tmp_path, n_patients=24)
    import uq.evaluate as E
    E.SUBGROUP_MIN_N = 5
    cfg = _cfg(tmp_path)

    # 1. DICOM -> 16-bit PNG; MONOCHROME1 inverted so tissue is bright
    idx = dicom_to_png.convert_all(cfg, workers=1)
    assert idx["error"].isna().all()
    png = cv2.imread(idx["png_path"].iloc[0], cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint16 and png.max() == 65535
    assert png[80, 0 if "L" in idx["png_path"].iloc[0].split("/")[-1] else -1] > png[0, 64]

    # 2. exam table grouped by accession, C-view filtered, labels derived, patient-level split
    df = build_metadata.build(cfg)
    assert df.groupby("exam_id").size().eq(4).all()
    assert df["exam_id"].nunique() == 24 * 2
    per_patient_splits = df.groupby("patient_id")["split"].nunique()
    assert per_patient_splits.max() == 1
    assert set(df["split"]) == {"calib", "test", "finetune"}
    assert ((df["years_to_cancer"] < 100) == df["patient_id"].isin(
        df[df["years_to_cancer"] < 100]["patient_id"])).all()

    # 3. a fake "released" AsymMirai model (full pickle, as the real one is)
    (tmp_path / "weights").mkdir()
    orig = make_original()
    torch.save(orig, tmp_path / "weights/full.pt")
    # calibrate the fake model's asymmetry scaling so the sigmoid is not saturated
    from torch.utils.data import DataLoader
    from uq.dataset import ExamDataset, exams_wide
    from uq.models.loading import load_model
    from uq.finetune_head import init_asym_stats
    from uq.preprocessing import Preprocessor
    tmp_model = load_model(cfg)
    ex = exams_wide(pd.read_csv(cfg.path("exam_csv")))
    init_asym_stats(tmp_model, DataLoader(ExamDataset(ex, Preprocessor.from_config(cfg)), batch_size=8), "cpu")
    orig.learned_asym_mean.data.fill_(tmp_model.asym_mean.item())
    orig.learned_asym_std.data.fill_(tmp_model.asym_std.item())
    torch.save(orig, tmp_path / "weights/full.pt")

    # 4. fine-tune two heads with dropout -> ensemble members
    out_dir = cfg.path("outputs") / "finetuned_heads"; out_dir.mkdir(parents=True)
    heads = [finetune_head.train_one(cfg, s, epochs=1, lr=1e-3, horizon=3, device=get_device("cpu"),
                                     out_dir=out_dir, checkpoint=None, bootstrap=True) for s in (0, 1)]
    cfg["paths"]["weights"]["ensemble_members"] = [str(h) for h in heads]

    # 5. inference
    run_dir = inference.run(cfg, run_name="smoke", device=torch.device("cpu"))
    pred = pd.read_csv(run_dir / "predictions_clean.csv")
    assert len(pred) == 48
    for m in ("mc_dropout", "tta", "mc_tta", "ensemble"):
        assert pred[f"{m}_std"].gt(0).any(), m
    s = np.load(run_dir / "samples_clean.npz")
    assert s["mc_dropout_probs"].shape == (3, 48)
    info = json.loads((run_dir / "run_info.json").read_text())
    assert set(info["conditions"]) == {"clean", "gaussian_s1", "gaussian_s3"}

    # 6. evaluation
    ev = evaluate.run(cfg, run_dir)
    res = json.loads((ev / "metrics.json").read_text())
    assert "mc_tta" in res["methods"] and "conformal" in res["methods"]["mc_tta"]
    for f in ("roc.png", "reliability.png", "risk_coverage_auc.png", "uncertainty_hist.png",
              "conformal_sets.png", "robustness_uncertainty.png", "subgroup_referral.png"):
        img = cv2.imread(str(ev / f))
        assert img is not None and img.std() > 5, f"{f} is blank"
    assert (ev / "report.md").read_text().startswith("# UQ evaluation report")

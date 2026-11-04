"""Free-tier workflow: plan from tables -> enriched subset -> targeted download -> resized PNGs
-> chunked, resumable inference -> evaluation with the enrichment caveat."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_original
from synthetic import make
from uq.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class FakeS3:
    """Serves files from a local folder laid out like the EMBED bucket."""

    def __init__(self, store: Path, strip_images: bool = False):
        self.store, self.strip, self.calls = store, strip_images, []

    def download_file(self, bucket, key, dst):
        self.calls.append(key)
        if self.strip and key.startswith("images/"):
            raise FileNotFoundError("NoSuchKey")  # this layout has no images/ root
        src = self.store / key
        if not src.exists():
            raise FileNotFoundError(key)
        shutil.copy(src, dst)


def _cfg(root):
    return load_config(ROOT / "configs" / "free_tier.yaml", overrides={
        "root": str(root),
        "paths": {"weights": {"asymmirai_full_model": "weights/full.pt"}},
        "preprocessing": {"target_size": [320, 256]},
        "labels": {"require_followup_for_negatives": False, "horizons": [1, 2, 3]},
        "subset": {"horizon": 3, "controls_per_case": 2, "max_exams": 18},
        "conformal": {"horizon": 3, "alpha": 0.2},
        "uq": {"mc_samples": 3, "tta_samples": 2, "num_workers": 0, "chunk_size": 4,
               "conditions": [{"name": "clean"}, {"name": "gaussian", "severity": 1}]},
    })


@pytest.fixture
def planned(tmp_path):
    make(tmp_path, n_patients=30, cancer_frac=0.4)
    store = tmp_path / "s3store"
    shutil.move(str(tmp_path / "data/images"), str(store / "images"))  # images now only "in the bucket"
    cfg = _cfg(tmp_path)
    from uq.data import build_metadata, select_subset
    plan = build_metadata.build(cfg, plan=True)
    sub = select_subset.run(cfg)
    return cfg, store, plan, sub


def test_plan_and_enriched_subset(planned):
    cfg, _, plan, sub = planned
    assert plan["exam_id"].nunique() == 60 and "anon_dicom_path" in plan
    ex = sub.drop_duplicates("exam_id")
    cases = ex[ex["years_to_cancer"] <= 3]
    controls = ex[ex["years_to_cancer"] >= 100]
    assert len(ex) <= 18 + 1 and 5 <= len(cases) <= 7          # n_cases = 18 // (1 + 2), split quotas rounded
    assert len(cases) <= len(controls) <= 2 * len(cases)        # up to 2 controls per case (pool permitting)
    assert set(cases["split"]) == {"calib", "test"}             # both splits get cases
    assert ex["patient_id"].is_unique                               # one exam per patient
    assert set(controls["split"]) <= set(cases["split"])            # controls drawn from the cases' splits
    assert sub["enriched"].all() and sub.groupby("exam_id").size().eq(4).all()


def test_download_only_subset_with_key_fallback(planned):
    from uq.data.download_embed import download_exams
    cfg, store, _, sub = planned
    stats = download_exams(cfg, workers=2, s3=FakeS3(store))
    assert stats == {"downloaded": 4 * sub["exam_id"].nunique(), "skipped": 0, "failed": 0}
    assert len(list(cfg.path("dicom_dir").rglob("*.dcm"))) == len(sub)
    assert download_exams(cfg, workers=2, s3=FakeS3(store))["skipped"] == len(sub)   # resumable
    # bucket layout without the images/ root -> second key form is used
    shutil.rmtree(cfg.path("dicom_dir"))
    stripped = store.parent / "s3store_stripped"; shutil.copytree(store / "images", stripped)
    assert download_exams(cfg, workers=2, s3=FakeS3(stripped, strip_images=True))["failed"] == 0


def test_free_tier_end_to_end_with_resume(no_cuda, planned):
    from uq import evaluate, inference
    from uq.data.dicom_to_png import convert_all
    from uq.data.download_embed import download_exams
    cfg, store, _, sub = planned

    download_exams(cfg, workers=2, s3=FakeS3(store))
    idx = convert_all(cfg, workers=1)                                # free tier: resize + delete DICOMs
    assert idx["error"].isna().all()
    assert not list(cfg.path("dicom_dir").rglob("*.dcm"))             # DICOMs deleted after conversion
    png = cv2.imread(idx["png_path"].iloc[0], cv2.IMREAD_UNCHANGED)
    assert png.dtype == np.uint16 and png.shape == (320, 256)        # stored at model input size
    assert all(Path(p).exists() for p in sub["file_path"])           # planned paths == converted paths
    # new session after a disconnect: DICOMs are gone but PNGs exist -> nothing is re-downloaded
    assert download_exams(cfg, workers=2, s3=FakeS3(store))["skipped"] == len(sub)

    (cfg.path("weights", "asymmirai_full_model").parent).mkdir(parents=True)
    torch.save(make_original(), cfg.path("weights", "asymmirai_full_model"))

    run_dir = inference.run(cfg, device=torch.device("cpu"))          # run_name comes from the config
    assert run_dir.name == "free_run"
    first = pd.read_csv(run_dir / "predictions_clean.csv")
    n_chunks = len(list((run_dir / "chunks" / "clean").glob("chunk_*.npz")))
    assert n_chunks == int(np.ceil(18 / 4))

    # simulate a disconnect: lose one finished chunk and the merged outputs, then rerun
    (run_dir / "chunks" / "clean" / "chunk_0002.npz").unlink()
    (run_dir / "predictions_clean.csv").unlink()
    inference.run(cfg, device=torch.device("cpu"))
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["conditions"]["clean"]["chunks_computed"] == 1
    assert info["conditions"]["clean"]["chunks_resumed"] == n_chunks - 1
    assert info["conditions"]["clean"]["seconds_per_exam"] > 0
    second = pd.read_csv(run_dir / "predictions_clean.csv")
    pd.testing.assert_frame_equal(first, second)                     # resumed == uninterrupted

    # a missing PNG is skipped, not fatal
    Path(sub["file_path"].iloc[0]).unlink()
    other = inference.run(cfg, run_name="missing_png", device=torch.device("cpu"))
    assert json.loads((other / "run_info.json").read_text())["n_exams_missing_images"] == 1

    ev = evaluate.run(cfg, run_dir)
    assert "enriched" in (ev / "report.md").read_text()

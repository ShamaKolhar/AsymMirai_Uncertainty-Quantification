"""Synthetic EMBED-like data for end-to-end tests (no real patient data needed).

Creates 16-bit DICOMs (some MONOCHROME1), an EMBED-style metadata table and clinical table.
"Cancer" patients get a bright lesion in ONE breast -> real asymmetry signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_dicom(path: Path, arr: np.ndarray, monochrome1: bool, lat: str, view: str):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.ImageLaterality, ds.ViewPosition = lat, view
    stored = (4095 - arr) if monochrome1 else arr
    ds.PixelData = stored.astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def breast(h, w, lat, rng, lesion=False):
    yy, xx = np.mgrid[0:h, 0:w]
    cx = 0 if lat == "L" else w - 1
    r = np.sqrt(((xx - cx) / (0.8 * w)) ** 2 + ((yy - h / 2) / (0.45 * h)) ** 2)
    img = np.where(r < 1, 1500 + 800 * (1 - r) + rng.normal(0, 60, (h, w)), 0)
    if lesion:
        ly, lx = rng.integers(h // 3, 2 * h // 3), (w // 4 if lat == "L" else 3 * w // 4)
        img += 2200 * np.exp(-(((yy - ly) ** 2 + (xx - lx) ** 2) / (2 * (0.06 * h) ** 2)))
    return np.clip(img, 0, 4095)


def make(root: Path, n_patients: int = 30, exams_per_patient: int = 2, cancer_frac: float = 0.35,
         h: int = 160, w: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    meta_rows, clin_rows = [], []
    races = ["White", "Black", "Asian"]
    for pi in range(n_patients):
        empi = 10_000_000 + pi
        cancer = rng.random() < cancer_frac
        cancer_side = rng.choice(["L", "R"])
        start = pd.Timestamp("2014-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
        dates = pd.date_range(start, periods=exams_per_patient + 1, freq="365D")
        for ei in range(exams_per_patient):
            acc = 5_000_000_000 + pi * 10 + ei
            sdate = dates[ei]
            for lat in ("L", "R"):
                for view in ("CC", "MLO"):
                    rel = f"{empi}/1.2.{acc}/1.3.{acc}{lat}{view}/1.4.{acc}{lat}{view}.dcm"
                    arr = breast(h, w, lat, rng, lesion=cancer and lat == cancer_side)
                    _write_dicom(root / "data/images" / rel, arr, monochrome1=bool(rng.random() < 0.2), lat=lat, view=view)
                    meta_rows.append({"anon_dicom_path": "images/cohort_1/" + rel, "empi_anon": empi, "acc_anon": acc,
                                      "study_date_anon": str(sdate), "ImageLateralityFinal": lat, "ViewPosition": view,
                                      "FinalImageType": "2D", "spot_mag": np.nan})
            # also a C-view duplicate that must be filtered out
            meta_rows.append({"anon_dicom_path": f"images/cohort_1/{empi}/fake_cview_{acc}.dcm", "empi_anon": empi,
                              "acc_anon": acc, "study_date_anon": str(sdate), "ImageLateralityFinal": "L",
                              "ViewPosition": "CC", "FinalImageType": "cview", "spot_mag": np.nan})
            clin_rows.append({"empi_anon": empi, "acc_anon": acc, "study_date_anon": sdate.date().isoformat(),
                              "screen_exam": True, "desc": "MG Screening Bilateral", "path_severity": np.nan,
                              "procdate_anon": np.nan, "age_at_study": float(rng.integers(40, 80)),
                              "race": races[pi % 3], "tissueden": float(rng.integers(1, 5))})
        # final diagnostic exam (cancer diagnosis or a negative follow-up)
        clin_rows.append({"empi_anon": empi, "acc_anon": 9_000_000_000 + pi, "study_date_anon": dates[-1].date().isoformat(),
                          "screen_exam": False, "desc": "MG Diagnostic", "path_severity": 0.0 if cancer else np.nan,
                          "procdate_anon": dates[-1].date().isoformat() if cancer else np.nan,
                          "age_at_study": 60.0, "race": races[pi % 3], "tissueden": 2.0})
    tables = root / "data/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta_rows).to_csv(tables / "EMBED_OpenData_NUS_Datathon_metadata_reduced.csv", index=False)
    pd.DataFrame(clin_rows).to_csv(tables / "EMBED_OpenData_NUS_Datathon_clinical_reduced.csv", index=False)
    return root

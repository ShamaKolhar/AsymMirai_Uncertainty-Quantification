"""Convert EMBED DICOMs to 16-bit PNGs in the format Mirai / AsymMirai expect.

Design:
  * 16-bit output. The model normalises with 16-bit statistics (mean 7699.5, std 11765.06),
    so 8-bit images would collapse to a near-constant input.
  * Mirai's convention is DCMTK `dcmj2pnm +on2 --min-max-window`: a min-max window
    mapped to the full 16-bit range. This is reproduced in pure Python (or use --use-dcmtk).
  * MONOCHROME1 images (bright = low value) are inverted so tissue is always bright.
  * Output paths mirror the DICOM's relative path so they join back to the EMBED metadata
    table (`anon_dicom_path`) with a merge, and an index CSV (dicom -> png) is written.

Usage:
    python -m uq.data.dicom_to_png --config configs/uq_config.yaml [--workers 4] [--use-dcmtk]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from uq.config import load_config


def dicom_to_uint16(ds) -> np.ndarray:
    """Pixel array -> uint16 using a min-max window over the full 16-bit range (dcmj2pnm +on2 --min-max-window)."""
    img = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    img = img * slope + intercept
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint16)
    img = (img - lo) / (hi - lo)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        img = 1.0 - img
    return np.round(img * 65535.0).astype(np.uint16)


def png_path_for(dicom_path: Path, dicom_root: Path, png_root: Path) -> Path:
    rel = dicom_path.relative_to(dicom_root)
    return (png_root / rel).with_suffix(".png")


def _convert_one(args) -> tuple[str, str, str | None]:
    dcm_path, dicom_root, png_root, use_dcmtk, overwrite = args
    import pydicom

    dcm_path, dicom_root, png_root = Path(dcm_path), Path(dicom_root), Path(png_root)
    out = png_path_for(dcm_path, dicom_root, png_root)
    if out.exists() and not overwrite:
        return str(dcm_path), str(out), None
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if use_dcmtk:
            subprocess.run(["dcmj2pnm", "+on2", "--min-max-window", str(dcm_path), str(out)],
                           check=True, capture_output=True)
        else:
            ds = pydicom.dcmread(str(dcm_path))
            arr = dicom_to_uint16(ds)
            if not cv2.imwrite(str(out), arr):
                raise IOError("cv2.imwrite failed")
        return str(dcm_path), str(out), None
    except Exception as e:  # keep going, report at the end
        return str(dcm_path), "", f"{type(e).__name__}: {e}"


def convert_all(cfg, workers: int = 4, use_dcmtk: bool = False, overwrite: bool = False) -> pd.DataFrame:
    dicom_root, png_root = cfg.path("dicom_dir"), cfg.path("png_dir")
    if use_dcmtk and shutil.which("dcmj2pnm") is None:
        raise RuntimeError("dcmj2pnm not found; install dcmtk (apt-get install dcmtk) or drop --use-dcmtk")
    paths = sorted(dicom_root.rglob("*.dcm"))
    print(f"Found {len(paths)} DICOMs under {dicom_root}")
    jobs = [(str(p), str(dicom_root), str(png_root), use_dcmtk, overwrite) for p in paths]
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in tqdm(ex.map(_convert_one, jobs, chunksize=16), total=len(jobs)):
                rows.append(r)
    else:
        rows = [_convert_one(j) for j in tqdm(jobs)]
    df = pd.DataFrame(rows, columns=["dicom_path", "png_path", "error"])
    # EMBED's metadata stores paths like images/cohort_1/<empi>/<study>/<series>/<sop>.dcm
    prefix = cfg["download"]["prefix"]
    df["anon_dicom_path"] = [prefix + str(Path(p).relative_to(dicom_root)) for p in df["dicom_path"]]
    out_csv = cfg.path("image_index_csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_err = df["error"].notna().sum()
    print(f"Converted {len(df) - n_err} / {len(df)} files. Index written to {out_csv}")
    if n_err:
        print(df[df["error"].notna()].head(10).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-dcmtk", action="store_true", help="use DCMTK dcmj2pnm exactly like Mirai")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    convert_all(load_config(args.config), args.workers, args.use_dcmtk, args.overwrite)


if __name__ == "__main__":
    main()

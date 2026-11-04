"""Download EMBED DICOMs from S3.

Credentials are NEVER stored in code. They are read from the standard AWS
environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), an AWS
profile, or - on Colab - from Colab Secrets (key names as above).

Usage:
    python -m uq.data.download_embed --config configs/uq_config.yaml [--max-files 200]   # everything under prefix
    python -m uq.data.download_embed --config configs/free_tier.yaml --exam-csv           # only the chosen exams
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from uq.config import load_config


def _load_colab_secrets() -> None:
    """If running on Colab and the env vars are missing, pull them from Colab Secrets."""
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return
    try:
        from google.colab import userdata  # type: ignore

        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            val = userdata.get(key)
            if val:
                os.environ[key] = val
    except Exception:
        pass


def local_path_for(cfg, key: str) -> Path:
    """Local DICOM path for an S3 key / EMBED anon_dicom_path: dicom_dir / (key minus path_root)."""
    root = cfg["download"].get("path_root", "images/")
    rel = key[len(root):] if key.startswith(root) else key
    return cfg.path("dicom_dir") / rel


def _client(cfg, s3=None):
    if s3 is not None:
        return s3
    import boto3

    _load_colab_secrets()
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        raise RuntimeError(
            "No AWS credentials found. Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
            "(or add them to Colab Secrets). Never paste keys into notebooks or READMEs."
        )
    return boto3.client("s3", region_name=cfg["download"]["region"])


def download(cfg, max_files: int | None = None, dry_run: bool = False, s3=None) -> int:
    """Download everything under download.prefix (resumable)."""
    dl = cfg["download"]
    max_files = max_files if max_files is not None else dl.get("max_files")
    s3 = _client(cfg, s3)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=dl["bucket"], Prefix=dl["prefix"]):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local = local_path_for(cfg, key)
            if local.exists() and local.stat().st_size == obj["Size"]:
                n += 1
                continue  # resumable
            if not dry_run:
                local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(dl["bucket"], key, str(local))
            n += 1
            if max_files and n >= max_files:
                return n
    return n


def download_exams(cfg, exam_csv=None, workers: int = 8, s3=None) -> dict:
    """Download ONLY the DICOMs of the exams in `exam_csv` (default: paths.exam_csv). Resumable.

    Tries the EMBED path as the S3 key first and, if missing, the key without the path root
    (bucket layouts have differed between EMBED releases)."""
    import pandas as pd

    df = pd.read_csv(exam_csv or cfg.path("exam_csv"))
    if "anon_dicom_path" not in df:
        raise ValueError("exam CSV has no anon_dicom_path column - build it with build_metadata --plan")
    keys = sorted(df["anon_dicom_path"].dropna().unique())
    png_for = dict(zip(df["anon_dicom_path"], df["file_path"])) if "file_path" in df else {}
    s3 = _client(cfg, s3)
    bucket = cfg["download"]["bucket"]
    root = cfg["download"].get("path_root", "images/")

    def fetch(key):
        local = local_path_for(cfg, key)
        png = png_for.get(key)
        if (local.exists() and local.stat().st_size > 0) or (isinstance(png, str) and Path(png).exists()):
            return "skipped"  # already downloaded, or already converted in an earlier session
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        for k in dict.fromkeys([key, key[len(root):] if key.startswith(root) else key]):
            try:
                s3.download_file(bucket, k, str(tmp))
                tmp.rename(local)  # only complete files get the real name -> safe to resume
                return "downloaded"
            except Exception as e:  # try the next key form
                err = e
        return f"failed: {err}"

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for key, r in zip(keys, tqdm(ex.map(fetch, keys), total=len(keys))):
            if r.startswith("failed"):
                stats["failed"] += 1; failures.append((key, r))
            else:
                stats[r] += 1
    print(f"{len(keys)} DICOMs for {df['exam_id'].nunique()} exams: {stats}")
    for k, r in failures[:5]:
        print("  ", k, r)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exam-csv", nargs="?", const="__config__", default=None,
                    help="download only the exams in this CSV (no value = paths.exam_csv)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.exam_csv:
        download_exams(cfg, None if args.exam_csv == "__config__" else args.exam_csv, args.workers)
        return
    n = download(cfg, args.max_files, args.dry_run)
    print(f"Done. {n} files present in {cfg.path('dicom_dir')}")


if __name__ == "__main__":
    main()

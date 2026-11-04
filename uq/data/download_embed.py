"""Download EMBED DICOMs from S3.

Credentials are NEVER stored in code. They are read from the standard AWS
environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), an AWS
profile, or - on Colab - from Colab Secrets (key names as above).

Usage:
    python -m uq.data.download_embed --config configs/uq_config.yaml [--max-files 200]
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


def download(cfg, max_files: int | None = None, dry_run: bool = False) -> int:
    import boto3

    _load_colab_secrets()
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        raise RuntimeError(
            "No AWS credentials found. Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
            "(or add them to Colab Secrets). Never paste keys into notebooks or READMEs."
        )
    dl = cfg["download"]
    max_files = max_files if max_files is not None else dl.get("max_files")
    out_root = cfg.path("dicom_dir")
    s3 = boto3.client("s3", region_name=dl["region"])
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=dl["bucket"], Prefix=dl["prefix"]):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local = out_root / key[len(dl["prefix"]):]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = download(cfg, args.max_files, args.dry_run)
    print(f"Done. {n} files present in {cfg.path('dicom_dir')}")


if __name__ == "__main__":
    main()

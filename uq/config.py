"""Config loading, path resolution, seeding and device selection."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "uq_config.yaml"


class Config(dict):
    """dict with attribute access and path helpers."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    @property
    def root_path(self) -> Path:
        root = os.environ.get("ASYMMIRAI_UQ_ROOT", self.get("root", "."))
        root = Path(root)
        return root if root.is_absolute() else (REPO_ROOT / root).resolve()

    def path(self, *keys: str) -> Path:
        """Resolve a path from the `paths` section, e.g. cfg.path('weights', 'asymmirai_full_model')."""
        node: Any = self["paths"]
        for k in keys:
            node = node[k]
        if node is None:
            raise ValueError(f"paths.{'.'.join(keys)} is not set in the config")
        p = Path(node)
        return p if p.is_absolute() else self.root_path / p


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return Config(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:  # pragma: no cover
        pass


def get_device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

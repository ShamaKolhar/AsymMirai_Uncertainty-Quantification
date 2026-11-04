"""Image loading, normalisation, corruptions (robustness conditions) and paired TTA.

Design:
  * Images are read with IMREAD_UNCHANGED (16-bit). IMREAD_GRAYSCALE would silently convert
    to 8-bit, so an 8-bit PNG raises an error instead of producing a flat image.
  * Corruptions are a separate experimental condition (clean is always evaluated too),
    scaled to the 16-bit range, with deterministic per-exam seeds so every UQ method
    sees exactly the same corrupted image.
  * TTA augmentations are applied identically to the left and right image of a pair:
    independent augmentation would create artificial asymmetry in an asymmetry model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MAX16 = 65535.0


class BitDepthError(ValueError):
    pass


def load_png(path: str, expected_bit_depth: int = 16) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if expected_bit_depth == 16 and img.dtype != np.uint16:
        raise BitDepthError(
            f"{path} is {img.dtype}, but the model expects 16-bit PNGs (mean 7699.5 / std 11765). "
            "Re-run uq.data.dicom_to_png, or set preprocessing.expected_bit_depth: 8 and provide 8-bit stats."
        )
    return img.astype(np.float32)


# ----------------------------------------------------------------------------- corruptions
GAUSSIAN_SIGMA = {1: 0.01, 2: 0.03, 3: 0.06, 4: 0.10, 5: 0.15}  # fraction of full range
BLUR_KERNEL = {1: 3, 2: 7, 3: 11, 4: 15, 5: 21}
SALT_PEPPER_P = {1: 0.002, 2: 0.01, 3: 0.03, 4: 0.05, 5: 0.1}


def corrupt(img: np.ndarray, name: str, severity: int, rng: np.random.Generator, max_val: float = MAX16) -> np.ndarray:
    """Apply a corruption in raw pixel space (before normalisation). Background (0) stays 0 for noise."""
    if name in (None, "clean"):
        return img
    out = img.copy()
    if name == "gaussian":
        sigma = GAUSSIAN_SIGMA[severity] * max_val
        out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    elif name == "blur":
        k = BLUR_KERNEL[severity]
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif name == "salt_pepper":
        p = SALT_PEPPER_P[severity]
        r = rng.random(out.shape)
        out[r < p / 2] = 0
        out[r > 1 - p / 2] = max_val
    else:
        raise ValueError(f"unknown corruption {name}")
    return np.clip(out, 0, max_val)


# ----------------------------------------------------------------------------- normalise + resize
@dataclass
class Preprocessor:
    img_mean: float = 7699.5
    img_std: float = 11765.06
    target_size: tuple = (1664, 2048)
    expected_bit_depth: int = 16

    @classmethod
    def from_config(cls, cfg) -> "Preprocessor":
        p = cfg["preprocessing"]
        mean, std = p["img_mean"], p["img_std"]
        if p.get("expected_bit_depth", 16) == 8:
            if p.get("img_mean_8bit") is None or p.get("img_std_8bit") is None:
                raise ValueError("8-bit mode needs preprocessing.img_mean_8bit / img_std_8bit "
                                 "(compute them with uq.preprocessing.compute_stats)")
            mean, std = p["img_mean_8bit"], p["img_std_8bit"]
        return cls(mean, std, tuple(p["target_size"]), p.get("expected_bit_depth", 16))

    @property
    def max_val(self) -> float:
        return MAX16 if self.expected_bit_depth == 16 else 255.0

    def blank(self) -> torch.Tensor:
        return torch.zeros(3, *self.target_size)

    def __call__(self, path, corruption: str | None = None, severity: int = 0,
                 rng: np.random.Generator | None = None) -> torch.Tensor:
        if path is None or (isinstance(path, float) and math.isnan(path)):
            return self.blank()  # model treats an all-zero image as a missing view
        img = load_png(path, self.expected_bit_depth)
        if corruption and corruption != "clean":
            img = corrupt(img, corruption, severity, rng or np.random.default_rng(0), self.max_val)
        t = torch.from_numpy((img - self.img_mean) / self.img_std).float()
        t = t.expand(1, 3, *t.shape)
        return F.interpolate(t, size=self.target_size, mode="bilinear", align_corners=False)[0]


def compute_stats(paths, expected_bit_depth: int = 16, max_images: int = 500) -> tuple[float, float]:
    """Pixel mean/std over a sample of images (use for 8-bit mode or a new site)."""
    s = ss = n = 0.0
    for p in list(paths)[:max_images]:
        a = load_png(p, expected_bit_depth).astype(np.float64)
        s += a.sum(); ss += (a ** 2).sum(); n += a.size
    mean = s / n
    return float(mean), float(math.sqrt(ss / n - mean ** 2))


# ----------------------------------------------------------------------------- paired TTA
@dataclass
class PairedTTA:
    """Small geometric + intensity jitter applied identically to both breasts of a view pair."""
    max_rotation_deg: float = 5.0
    max_translate_frac: float = 0.03
    contrast_jitter: float = 0.05
    paired: bool = True

    @classmethod
    def from_config(cls, cfg) -> "PairedTTA":
        return cls(**cfg["uq"]["tta"])

    def _sample(self, n: int, gen: torch.Generator, device):
        ang = (torch.rand(n, generator=gen) * 2 - 1) * math.radians(self.max_rotation_deg)
        tx = (torch.rand(n, generator=gen) * 2 - 1) * self.max_translate_frac * 2
        ty = (torch.rand(n, generator=gen) * 2 - 1) * self.max_translate_frac * 2
        c = 1 + (torch.rand(n, generator=gen) * 2 - 1) * self.contrast_jitter
        return ang.to(device), tx.to(device), ty.to(device), c.to(device)

    @staticmethod
    def _apply(x: torch.Tensor, ang, tx, ty, c, bg: torch.Tensor) -> torch.Tensor:
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([cos, -sin, tx], -1), torch.stack([sin, cos, ty], -1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        # pad with the background value (normalised 0), not with 0 in normalised space
        y = F.grid_sample(x - bg, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + bg
        return (y - bg) * c.view(-1, 1, 1, 1) + bg

    def __call__(self, left: torch.Tensor, right: torch.Tensor, gen: torch.Generator, bg_value: float):
        n = left.shape[0]
        bg = torch.tensor(bg_value, device=left.device)
        p = self._sample(n, gen, left.device)
        q = p if self.paired else self._sample(n, gen, left.device)
        l_missing = left.flatten(1).abs().sum(1) == 0
        r_missing = right.flatten(1).abs().sum(1) == 0
        l2, r2 = self._apply(left, *p, bg), self._apply(right, *q, bg)
        l2[l_missing] = 0; r2[r_missing] = 0  # keep "missing view" signal intact
        return l2, r2

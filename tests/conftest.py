import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "AsymMirai", ROOT / "AsymMirai" / "asymmetry_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def no_cuda(monkeypatch):
    """Let the ORIGINAL AsymMirai code (which calls .cuda() everywhere) run on CPU."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self, raising=True)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self, raising=True)
    yield


def tiny_backbone():
    import torchvision.models as tvm
    torch.manual_seed(0)
    m = tvm.resnet18(weights=None)
    return torch.nn.Sequential(*list(m.children())[:-2]).eval()


def make_original(no_bias=False):
    from asymmetry_metrics import hybrid_asymmetry
    from mirai_localized_dif_head import LocalizedDifModel
    torch.manual_seed(1)
    orig = LocalizedDifModel(asymmetry_metric=hybrid_asymmetry, embedding_channel=512,
                             embedding_model=tiny_backbone(), use_stretch=True, train_backbone=False,
                             flexible_asymmetry=True, use_stretch_matrix=True, use_bias=not no_bias,
                             initial_asym_mean=20.0, initial_asym_std=5.0)
    return orig.eval()

import torch

from conftest import make_original
from uq.models.loading import from_original_model, load_model, save_checkpoint, load_checkpoint
from uq.models.uq_asymmirai import UQAsymMirai


def _imgs(B=2, H=320, W=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, 3, H, W, generator=g) for _ in range(4)]


def test_matches_original_forward(no_cuda):
    orig = make_original()
    uq = from_original_model(orig).eval()
    x = _imgs()
    with torch.no_grad():
        o, _ = orig(*x)
        u = uq(*x)
    assert torch.allclose(o[:, 1], u["prob"], atol=1e-6), (o[:, 1], u["prob"])


def test_loads_pickled_full_model(no_cuda, tmp_path):
    """Released weights are a torch.save(model) pickle -> must round-trip through load_model."""
    orig = make_original()
    p = tmp_path / "full.pt"
    torch.save(orig, p)
    from uq.config import load_config
    cfg = load_config(overrides={"root": str(tmp_path), "paths": {"weights": {"asymmirai_full_model": "full.pt"}}})
    uq = load_model(cfg).eval()
    x = _imgs()
    with torch.no_grad():
        assert torch.allclose(orig(*x)[0][:, 1], uq(*x)["prob"], atol=1e-6)


def test_missing_view_handled(no_cuda):
    uq = from_original_model(make_original()).eval()
    x = _imgs()
    x[2][0] = 0  # first exam has no left MLO
    with torch.no_grad():
        out = uq(*x)
    assert torch.isfinite(out["prob"]).all()


def test_paired_dropout_no_fake_asymmetry(no_cuda):
    """Identical left/right images must give zero asymmetry even with MC dropout on."""
    uq = from_original_model(make_original(), head_dropout=0.5).mc(True)
    x = _imgs(B=1)
    x[1] = x[0].clone(); x[3] = x[2].clone()
    with torch.no_grad():
        raws = [uq(*x)["raw_asymmetry"].item() for _ in range(5)]
    assert max(abs(r) for r in raws) < 1e-4


def test_mc_dropout_on_cached_features_varies(no_cuda):
    uq = from_original_model(make_original(), head_dropout=0.3).mc(True)
    x = _imgs()
    with torch.no_grad():
        f = uq.features(*x)
        ps = torch.stack([uq.head(f)["prob"] for _ in range(10)])
    assert ps.std(0).min() > 0


def test_mc_off_is_deterministic(no_cuda):
    uq = from_original_model(make_original(), head_dropout=0.3).mc(False)
    x = _imgs()
    with torch.no_grad():
        a, b = uq(*x)["prob"], uq(*x)["prob"]
    assert torch.equal(a, b)


def test_checkpoint_roundtrip(no_cuda, tmp_path):
    uq = from_original_model(make_original())
    p = tmp_path / "ck.pt"
    save_checkpoint(uq, p)
    uq2 = load_checkpoint(p, backbone=uq.backbone)
    x = _imgs()
    with torch.no_grad():
        assert torch.allclose(uq.eval()(*x)["prob"], uq2.eval()(*x)["prob"])


def test_convnext_backbone_builds():
    from uq.models.uq_asymmirai import build_backbone
    bb, ch = build_backbone("convnext_tiny", pretrained=False)
    m = UQAsymMirai(bb, embedding_channel=ch, backbone_name="convnext_tiny", head_dropout=0.1).eval()
    with torch.no_grad():
        out = m(*_imgs(B=1, H=320, W=320))
    assert out["prob"].shape == (1,)

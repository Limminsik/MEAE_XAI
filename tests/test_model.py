"""S2 단위 테스트 (RESEARCH_DESIGN.md §11)."""
import copy

import pytest
import torch

from src.data.build import load_cfg
from src.model import losses, meae

CFG = load_cfg()
PAD_EACH = CFG["data"]["pad_each"]
SEG_LEN = CFG["data"]["fs"] * CFG["data"]["seg_sec"]
PAD_TO = CFG["data"]["pad_to"]


# n_encoders 는 K 후보 목록이 아니라 확정된 단일 값(8)이다.
@pytest.fixture(params=[CFG["model"]["n_encoders"]])
def model(request):
    torch.manual_seed(0)
    return meae.build(CFG, request.param).eval()


def test_pad_crop_roundtrip():
    """3600 → 3840 → 3600 이 원본과 동일하고, 패딩 구간이 실제로 0."""
    x = torch.randn(2, 1, SEG_LEN)
    p = meae.pad(x, PAD_EACH)
    assert p.shape == (2, 1, PAD_TO)
    assert (p[..., :PAD_EACH] == 0).all() and (p[..., -PAD_EACH:] == 0).all()
    assert torch.equal(meae.crop(p, PAD_EACH), x)


def test_shapes(model):
    """component·masked_reconstruct 출력이 (B,1,3840), 크롭 후 (B,1,3600)."""
    x = meae.pad(torch.randn(2, 1, SEG_LEN), PAD_EACH)
    y, zs = model(x)
    assert y.shape == x.shape
    assert len(zs) == model.n_encoders
    assert zs[0].shape[-1] == PAD_TO // meae.DOWNSAMPLE_FACTOR
    assert zs[0].shape[1] == CFG["model"]["hidden"] // model.n_encoders
    for out in (model.component(x, 0), model.masked_reconstruct(x, 0),
                model.masked_reconstruct(x, [0, 1])):
        assert out.shape == (2, 1, PAD_TO)
        assert meae.crop(out, PAD_EACH).shape == (2, 1, SEG_LEN)


def test_length_must_be_multiple_of_256():
    """3600을 그대로 넣으면 만들 때 막아야 한다 (재구성 길이가 3584로 어긋남)."""
    with pytest.raises(ValueError, match="256"):
        meae.MEAE(n_encoders=8, input_length=SEG_LEN)


def test_k6_rejected():
    """K=6은 GroupNorm 제약으로 불가 — 조용히 깨지지 않고 명시적으로 막는다."""
    with pytest.raises(ValueError, match="나눠지지 않는다"):
        meae.MEAE(n_encoders=6, input_length=PAD_TO)


def test_mask_effect(model):
    """마스킹은 인코딩만 0으로 바꾸고 인코더 가중치는 건드리지 않는다 (§0 원칙 1)."""
    x = meae.pad(torch.randn(2, 1, SEG_LEN), PAD_EACH)
    before = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    zs = model.encode(x)
    masked = model._mask(zs, [0])
    assert torch.equal(masked[0], zs[0])                      # 유지된 인코딩은 그대로
    for z in masked[1:]:
        assert torch.count_nonzero(z) == 0                    # 나머지는 전부 0
    for i, z in enumerate(zs[1:], 1):
        assert torch.equal(z, model.encode(x)[i])             # 원본 인코딩은 불변

    model.component(x, 0)
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"가중치가 변했다: {k}"


def test_component_equals_manual_mask(model):
    """component(x,k) == 직접 마스킹한 decode 결과."""
    x = meae.pad(torch.randn(2, 1, SEG_LEN), PAD_EACH)
    zs = model.encode(x)
    for k in range(model.n_encoders):
        manual = model.decode([z if i == k else torch.zeros_like(z) for i, z in enumerate(zs)])
        assert torch.allclose(model.component(x, k), manual, atol=1e-6)


def test_masked_all_equals_zero_encoding(model):
    """전부 마스킹하면 전영 인코딩 디코딩과 같아야 한다 (zero recon 항의 전제)."""
    x = meae.pad(torch.randn(2, 1, SEG_LEN), PAD_EACH)
    allmask = model.masked_reconstruct(x, list(range(model.n_encoders)))
    zeros = model.zero_encoding(2, PAD_TO // meae.DOWNSAMPLE_FACTOR, x.device)
    assert torch.allclose(allmask, model.decode(zeros), atol=1e-6)


def test_loss_terms(model):
    """4개 항이 모두 나오고 유한하며, total이 가중합과 일치."""
    crit = losses.build(CFG, model.n_encoders)
    x = meae.pad(torch.randn(2, 1, SEG_LEN), PAD_EACH)
    y, zs = model(x)
    d = crit(model, x, y, zs)
    assert set(d) == {"total", "recon", "mixing", "zero_recon", "z_l2"}
    for k, v in d.items():
        assert torch.isfinite(v), k
    expect = (d["recon"] + crit.lambda_z_l2 * d["z_l2"]
              + crit.lambda_mixing * d["mixing"]
              + crit.lambda_zero_recon * d["zero_recon"])
    assert torch.allclose(d["total"], expect, atol=1e-6)


def test_loss_is_mse_not_bce(model):
    """재구성 손실이 MSE인지 — 완전 재구성이면 0."""
    crit = losses.build(CFG, model.n_encoders)
    x = meae.pad(torch.randn(2, 1, SEG_LEN), PAD_EACH)
    assert crit.recon(x, x).item() == 0.0
    assert isinstance(crit.recon, torch.nn.MSELoss)

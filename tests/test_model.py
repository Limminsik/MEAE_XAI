"""모델 단위 테스트 — 확정 구조(C16)와 지도 손실, 잔차 마스킹 차단."""
import copy

import pytest
import torch

from src.data.build import load_cfg
from src.model import losses, meae

CFG = load_cfg()
SEG_LEN = CFG["data"]["fs"] * CFG["data"]["seg_sec"]
K = CFG["model"]["n_encoders"]
DEPTH = len(CFG["model"]["channels"])
FACTOR = 2 ** DEPTH


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return meae.build(CFG, K).eval()


# ---------------------------------------------------------------- 구조
def test_config_is_confirmed_c16():
    """config 기본값이 확정 모델 C16 과 일치하는가 — 재현의 전제다."""
    assert CFG["model"]["channels"] == [32, 32, 64, 64]
    assert CFG["model"]["dilations"] == [1, 1, 4, 32]
    assert CFG["model"]["hidden"] == 64 and K == 4
    assert CFG["model"]["skip_levels"] is None
    assert CFG["loss"]["gamma_sup"] == 50.0
    assert CFG["loss"]["gamma2_sup"] == 0.0 and CFG["loss"]["beta_sup"] == 0.0


def test_derived_geometry(model):
    """깊이 4 → 압축률 16 · 패딩 0 · 수용영역 3316(기준선과 동일)."""
    assert model.depth == DEPTH == 4
    assert model.downsample == FACTOR == 16
    assert model.pad_each == 0 and model.input_length == SEG_LEN
    rf, sec = meae.receptive_field(model.dilations, CFG["data"]["fs"])
    assert rf == 3316 and sec == pytest.approx(9.21, abs=0.01)


def test_fit_pad():
    """패딩 유도 — 깊이별 표와 일치, 항상 대칭."""
    assert meae.fit_pad(3600, 8) == (3840, 120)
    assert meae.fit_pad(3600, 6) == (3648, 24)
    assert meae.fit_pad(3600, 5) == (3616, 8)
    assert meae.fit_pad(3600, 4) == (3600, 0)


def test_pad_crop_roundtrip():
    """pad → crop 이 원본과 동일하고, 패딩 구간이 실제로 0."""
    length, pe = meae.fit_pad(SEG_LEN, 6)
    x = torch.randn(2, 1, SEG_LEN)
    p = meae.pad(x, pe)
    assert p.shape == (2, 1, length)
    assert (p[..., :pe] == 0).all() and (p[..., -pe:] == 0).all()
    assert torch.equal(meae.crop(p, pe), x)


def test_shapes(model):
    """인코딩·성분·복원의 모양이 유도값과 일치."""
    x = torch.randn(2, 1, SEG_LEN)
    y, zs = model(x)
    assert y.shape == x.shape
    assert len(zs) == K
    assert zs[0].shape[-1] == SEG_LEN // FACTOR          # 225
    assert zs[0].shape[1] == CFG["model"]["hidden"] // K
    for out in (model.component(x, 0), model.masked_reconstruct(x, 0),
                model.masked_reconstruct(x, [1, 2, 3])):
        assert out.shape == (2, 1, SEG_LEN)


def test_length_must_be_multiple_of_factor():
    """길이가 2^깊이의 배수가 아니면 만들 때 막는다."""
    with pytest.raises(ValueError, match=str(FACTOR)):
        meae.MEAE(n_encoders=K, input_length=SEG_LEN + 1,
                  channels=CFG["model"]["channels"])


def test_k6_rejected():
    """K=6은 GroupNorm 제약으로 불가 — 조용히 깨지지 않고 명시적으로 막는다."""
    with pytest.raises(ValueError, match="나눠지지 않는다"):
        meae.MEAE(n_encoders=6, input_length=SEG_LEN,
                  channels=CFG["model"]["channels"])


def test_dilation_default_matches_original():
    """dilations 전부 1 이면 수정 전 원본과 같은 연산이다 (파라미터 모양으로 확인)."""
    torch.manual_seed(0)
    a = meae.MEAE(n_encoders=K, input_length=SEG_LEN,
                  channels=CFG["model"]["channels"])          # dilations 없음
    torch.manual_seed(0)
    b = meae.MEAE(n_encoders=K, input_length=SEG_LEN,
                  channels=CFG["model"]["channels"], dilations=[1] * DEPTH)
    x = torch.randn(1, 1, SEG_LEN)
    with torch.no_grad():
        assert torch.equal(a(x)[0], b(x)[0])


# ---------------------------------------------------------------- 마스킹
def test_mask_effect(model):
    """마스킹은 인코딩만 0으로 바꾸고 인코더 가중치는 건드리지 않는다."""
    x = torch.randn(2, 1, SEG_LEN)
    before = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    zs = model.encode(x)
    masked = model._mask(zs, [0])
    assert torch.equal(masked[0], zs[0])                      # 유지된 인코딩은 그대로
    for z in masked[1:]:
        assert torch.count_nonzero(z) == 0                    # 나머지는 전부 0

    model.component(x, 0)
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"가중치가 변했다: {k}"


def test_component_equals_manual_mask(model):
    """component(x,k) == 직접 마스킹한 decode 결과 (잔차 없음이면 동일)."""
    x = torch.randn(2, 1, SEG_LEN)
    zs = model.encode(x)
    for k in range(K):
        manual = model.decode([z if i == k else torch.zeros_like(z)
                               for i, z in enumerate(zs)])
        assert torch.allclose(model.component(x, k), manual, atol=1e-6)


def test_b_equals_c(model):
    """B(잡음 3개 마스킹)와 C(심장만 남기기)는 K=4 에서 같은 연산이다."""
    x = torch.randn(2, 1, SEG_LEN)
    assert torch.allclose(model.masked_reconstruct(x, [1, 2, 3]),
                          model.component(x, 0), atol=1e-6)


# ---------------------------------------------------------------- 잔차 연결
def test_skip_masking_blocks_input():
    """인코딩·잔차를 모두 0으로 두면 출력이 입력에 의존하지 않는다 — 차단 검증."""
    torch.manual_seed(0)
    m = meae.MEAE(n_encoders=K, input_length=SEG_LEN,
                  channels=CFG["model"]["channels"], dilations=[1, 1, 4, 32],
                  skip_levels=[0], skip_weight=1.0).eval()
    x1, x2 = torch.randn(1, 1, SEG_LEN), torch.randn(1, 1, SEG_LEN) * 3
    outs = []
    with torch.no_grad():
        for x in (x1, x2):
            zs, sk = m.encode_all(x)
            z0 = [torch.zeros_like(z) for z in zs]
            s0 = [[torch.zeros_like(t) for t in s] for s in sk]
            outs.append(m.decode(z0, skips=s0))
    assert torch.equal(outs[0], outs[1])


def test_skip_component_ignores_other_encoders():
    """성분 k 는 다른 인코더의 잔차를 100배 흔들어도 변하지 않는다."""
    torch.manual_seed(0)
    m = meae.MEAE(n_encoders=K, input_length=SEG_LEN,
                  channels=CFG["model"]["channels"], dilations=[1, 1, 4, 32],
                  skip_levels=[0], skip_weight=1.0).eval()
    x = torch.randn(1, 1, SEG_LEN)
    with torch.no_grad():
        zs, sk = m.encode_all(x)
        base = m.decode(m._mask(zs, [0]), skips=m._mask_skips(sk, [0]))
        sk2 = [[t.clone() for t in q] for q in sk]
        sk2[1][0] = sk2[1][0] * 100
        pert = m.decode(m._mask(zs, [0]), skips=m._mask_skips(sk2, [0]))
    assert torch.equal(base, pert)


# ---------------------------------------------------------------- 지도 손실
def test_supervised_loss_terms(model):
    """항이 모두 나오고 유한하며, total = recon + λ_sup · sup."""
    crit = losses.build_supervised(CFG, K, model.pad_each)
    x = torch.randn(2, 1, SEG_LEN)
    y, zs = model(x)
    refs = torch.randn(2, K, SEG_LEN)
    d = crit(model, x, y, zs, refs)
    for key in ("total", "recon", "sup", "sup_wave", "sup_diff", "sup_diff2",
                "sup_freq") + tuple(f"sup_e{k+1}" for k in range(K)):
        assert key in d and torch.isfinite(d[key]), key
    assert torch.allclose(d["total"], d["recon"] + crit.lambda_sup * d["sup"],
                          atol=1e-6)


def test_supervised_loss_zero_when_perfect(model):
    """성분이 참조와 완전히 같으면 지도항이 0 이다."""
    crit = losses.build_supervised(CFG, K, model.pad_each)
    x = torch.randn(1, 1, SEG_LEN)
    y, zs = model(x)
    with torch.no_grad():
        refs = torch.stack([meae.crop(model.component(x, k), model.pad_each
                                      ).squeeze(1) for k in range(K)], dim=1)
    d = crit(model, x, y, zs, refs)
    assert d["sup"].item() == pytest.approx(0.0, abs=1e-10)


def test_supervise_count_must_match_k():
    """배정 수가 K 와 다르면 만들 때 막는다."""
    bad = {**CFG, "loss": {**CFG["loss"], "supervise": ["x_clean", "bw"]}}
    with pytest.raises(ValueError, match="배정"):
        losses.build_supervised(bad, K, 0)


def test_source_sigmas_scale():
    """σ_k 정규화 — 같은 오차라도 σ 가 작은 소스의 벌점이 커진다."""
    torch.manual_seed(0)
    m = meae.build(CFG, K).eval()
    crit0 = losses.build_supervised(CFG, K, m.pad_each)                    # σ 없음
    crit1 = losses.build_supervised(CFG, K, m.pad_each, sigmas=[2.0, 1.0, 1.0, 1.0])
    x = torch.randn(1, 1, SEG_LEN)
    y, zs = m(x)
    refs = torch.zeros(1, K, SEG_LEN)
    d0, d1 = crit0(m, x, y, zs, refs), crit1(m, x, y, zs, refs)
    # 첫 소스만 1/4 로 줄었으므로 전체 sup 은 줄어야 한다
    assert d1["sup"].item() < d0["sup"].item()

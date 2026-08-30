"""공용 원시 함수 — 번호 붙은 단계 스크립트(01~07)가 공유한다.

여기에는 **여러 단계가 함께 쓰는 것만** 둔다. 한 단계에서만 쓰는 계산·표·그림은
그 단계 스크립트 안에 둔다.

  · 체크포인트 로드          load_ckpt
  · 성분·재구성 추출         component_bank · reconstruct
  · 분절 내 표준화와 지표    znorm · pearson · aggregate · rmse_norm_matrix · mad_matrix
  · 표 렌더링                top_idx · mark · render

모든 계산은 **crop 후 중앙 3600 구간**에서 한다.
"""
import os

import numpy as np
import torch

from .data.dataset import REF_KEYS
from .model import meae
from .model.meae import enc_label

NOISE_REFS = ("bw", "ma", "em")


# ---------------------------------------------------------------- 모델·데이터
def load_ckpt(cfg, run):
    """run 은 실행 이름이거나 체크포인트 파일 경로다.
    후자를 허용하는 이유: pool/ 에 보관한 후보 에폭을 재학습 없이 그대로 불러 쓰기 위해서다."""
    if run.endswith(".pt") and os.path.exists(run):
        path = run
    else:
        name = os.path.basename(run)
        cand = [os.path.join("results", "02_model", run, f"{name}.pt"),
                os.path.join("_work", "archive", "runs", run, f"{name}.pt")]
        path = next((c for c in cand if os.path.exists(c)), cand[0])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # 체크포인트에 저장된 config를 우선한다 — hidden 등 구조 오버라이드가 반영돼 있다
    model = meae.build(ck.get("cfg", cfg), ck["n_encoders"])
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def component_bank(model, ds, device, idx, batch=100):
    """(n, K, 3600) 성분과 (n, R, 3600) 참조."""
    K, pad = model.n_encoders, model.pad_each
    comps, refs = [], []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        c = torch.stack([meae.crop(model.component(x, k), pad).squeeze(1)
                         for k in range(K)], 1)
        r = torch.stack([ds.ref_tensor(k, j).to(device).squeeze(1)
                         for k in REF_KEYS], 1)
        comps.append(c.cpu().numpy().astype(np.float64))
        refs.append(r.cpu().numpy().astype(np.float64))
    return np.concatenate(comps), np.concatenate(refs)


@torch.no_grad()
def reconstruct(model, ds, device, idx, batch=100):
    """(n, 3600) 입력과 마스킹 없는 재구성."""
    pad = model.pad_each
    rec = []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x = meae.pad(ds.tensor(j).to(device), pad)
        rec.append(meae.crop(model(x)[0], pad).squeeze(1).cpu().numpy().astype(np.float64))
    return ds.x_noisy[idx].astype(np.float64), np.concatenate(rec)


# ---------------------------------------------------------------- 표준화와 지표
def _center(a):
    return a - a.mean(-1, keepdims=True)


def znorm(a):
    """분절 내 표준화 — 마지막 축 기준 평균 0, 표준편차 1. 상수 신호는 0으로 둔다."""
    s = a.std(-1, keepdims=True)
    return np.where(s > 0, _center(a) / np.maximum(s, 1e-12), 0.0)


def pearson(comps, refs):
    """[S4-01] 1단계 — **분절 내** Pearson 상관. (n, K, R) 부호 있는 ρ.

    평균·표준편차는 해당 분절 안에서만 구한다. 분절을 이어붙여 일괄 계산하지 않는다 —
    분절마다 다른 잡음 구간이 주입되었고, 진폭 큰 분절이 결과를 지배하기 때문이다.
    상관 관련 코드는 전부 이 함수를 거친다.
    """
    zc, zr = znorm(comps), znorm(refs)
    return np.einsum("nkt,nrt->nkr", zc, zr) / zc.shape[-1]


def aggregate(rho):
    """[S4-01] 2단계 — 분절 간 집계. (ρ̄, σ, 양수비율) 각 (K, R).

      ρ̄ = mean_s |ρ|,  σ = std_s |ρ| (ddof=1)

    절댓값을 쓰는 이유: 인코딩과 디코더 가중치가 동시에 부호 반전되어도 재구성이 불변하므로
    성분이 참조와 반대 위상으로 수렴할 수 있다. 부호 반전은 무관이 아니라 반대 위상의 일치다.
    원 부호의 분포는 양수 비율로 따로 기록한다.
    """
    a = np.abs(rho)
    return a.mean(0), a.std(0, ddof=1), (rho > 0).mean(0)


def rmse_norm_matrix(comps, refs):
    """[S4-02] 정규화 RMSE. (n, K, R) 분절별 `sqrt(mean_i (ã−r̃)²)`.

    표준화하는 이유: 성분 진폭은 비선형 디코더의 임의 출력이고 참조 4종의 RMS도 서로 다르다
    (clean 0.204, 잡음 0.113~0.119 mV). 표준화하지 않으면 크기 차이가 값을 지배한다.
    부호 정렬을 하지 않으므로 반대 위상은 값이 커진다.
    """
    zc, zr = znorm(comps), znorm(refs)
    out = np.empty((zc.shape[0], zc.shape[1], zr.shape[1]))
    for k in range(zc.shape[1]):          # (n,K,R,T) 를 한 번에 만들면 수백 MB가 된다
        d = zc[:, k, None, :] - zr
        out[:, k, :] = np.sqrt((d ** 2).mean(-1))
    return out


def mad_matrix(comps, refs, with_argmax=False):
    """[S4-03] 국소 최대 편차. (n, K, R) 분절별 `max_i |ã−r̃|`. 단위는 표준편차.

    with_argmax=True 이면 최대 편차가 **어느 표본에서** 났는지도 함께 준다.
    """
    zc, zr = znorm(comps), znorm(refs)
    out = np.empty((zc.shape[0], zc.shape[1], zr.shape[1]))
    pos = np.empty(out.shape, dtype=np.int64) if with_argmax else None
    for k in range(zc.shape[1]):
        d = np.abs(zc[:, k, None, :] - zr)
        out[:, k, :] = d.max(-1)
        if with_argmax:
            pos[:, k, :] = d.argmax(-1)
    return (out, pos) if with_argmax else out


# ---------------------------------------------------------------- 표 렌더링
def top_idx(m, n=2, largest=True):
    """(K, R) 에서 **열별** 상·하위 n개 인덱스. 반환 (n, R), 1위가 첫 행.

    표시는 열 기준이다 — 한 참조를 어느 인코더가 가장 잘 잡는지를 열 안에서 비교한다.
    """
    return np.argsort(-m if largest else m, axis=0)[:n]


def mark(m, n=2, largest=True):
    """(K, R) → 열별 상·하위 n개 순위 배열. 0=미표시, 1=1위, 2=2위."""
    out = np.zeros(m.shape, dtype=int)
    for rank, ks in enumerate(top_idx(m, n, largest)):
        out[ks, np.arange(m.shape[1])] = rank + 1
    return out


def render(m, sd, flag, fmt="{:.3f}"):
    """CSV·콘솔용 표기. [1] = 열 1위, [2] = 열 2위 (굵은 글씨를 못 쓰므로)."""
    tag = {0: "", 1: " [1]", 2: " [2]"}
    return [[f"{fmt.format(m[k, r])}±{fmt.format(sd[k, r])}{tag[flag[k, r]]}"
             for r in range(m.shape[1])] for k in range(m.shape[0])]


def enc_names(K):
    return [enc_label(k) for k in range(K)]

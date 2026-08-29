"""S1 사전 단계 — 공개 데이터 내려받기·검증 (RESEARCH_DESIGN.md §4).

MIT-BIH Arrhythmia(mitdb)만 내려받는다. NSTDB와 GalaxyPPG는 이미 data/ 에 있다.
중단된 다운로드를 안전하게 이어받는다(불완전 기록만 다시 받음).
"""
import argparse
import os
import sys

import wfdb
import yaml

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
EXTS = (".dat", ".hea", ".atr")


def load_cfg(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def missing_records(dest):
    """세 확장자가 모두 있고 크기가 0이 아닌 기록만 완료로 본다."""
    out = []
    for r in MITDB_RECORDS:
        for e in EXTS:
            p = os.path.join(dest, r + e)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                out.append(r)
                break
    return out


def download_mitdb(dest):
    os.makedirs(dest, exist_ok=True)
    todo = missing_records(dest)
    if not todo:
        print(f"[skip] mitdb complete ({len(MITDB_RECORDS)} records): {dest}")
        return
    print(f"[dl] {len(todo)} records -> {dest}: {' '.join(todo)}")
    # 중단으로 남은 부분 파일 제거 후 재수신
    for r in todo:
        for e in EXTS:
            p = os.path.join(dest, r + e)
            if os.path.exists(p):
                os.remove(p)
    wfdb.dl_database("mitdb", dl_dir=dest, records=todo, overwrite=True)
    still = missing_records(dest)
    print(f"[dl] done. still missing: {still if still else 'none'}")


def verify(cfg):
    ok = True
    mitdb = cfg["paths"]["mitdb"]
    still = missing_records(mitdb)
    print(f"mitdb records: {len(MITDB_RECORDS) - len(still)}/{len(MITDB_RECORDS)}")
    if still:
        print(f"  MISSING: {still}")
        ok = False

    excl = set(cfg["data"]["mitdb_exclude"])
    lead = cfg["data"]["lead"]
    usable = [r for r in MITDB_RECORDS if r not in excl and r not in still]
    bad = [r for r in usable if lead not in wfdb.rdheader(os.path.join(mitdb, r)).sig_name]
    print(f"usable ({lead}, {sorted(excl)} 제외): {len(usable)}, {lead} 없는 기록: {bad}")
    ok &= not bad

    nstdb = cfg["paths"]["nstdb"]
    for n in ("bw", "ma", "em"):
        sig, fields = wfdb.rdsamp(os.path.join(nstdb, n))
        print(f"nstdb {n}: shape={sig.shape} fs={fields['fs']} units={fields['units']}")
        ok &= fields["fs"] == cfg["data"]["fs"]
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    cfg = load_cfg(a.config)
    if not a.verify_only:
        download_mitdb(cfg["paths"]["mitdb"])
    sys.exit(0 if verify(cfg) else 1)

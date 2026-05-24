"""공개 데이터베이스(PubChem)에서 약물 분자 기술자를 수집하여 CSV로 저장하는 모듈."""

import time
import os
from pathlib import Path

import requests
import pandas as pd

from params import DRUGS

# PubChem REST API 기본 URL
_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# 저장 경로
_DATA_DIR = Path(__file__).parent / "data"
_OUTPUT_CSV = _DATA_DIR / "drugs_descriptors.csv"


# =============================================================================
# 1단계: 약물명 → CID 조회
# =============================================================================

def fetch_cid(drug_name: str, timeout: int = 10) -> int:
    """PubChem에서 약물 이름으로 CID(화합물 식별자)를 조회한다.

    Args:
        drug_name: PubChem에 등록된 약물명 (예: 'ibuprofen')
        timeout: HTTP 요청 타임아웃 (초)

    Returns:
        PubChem CID (정수)

    Raises:
        ValueError: CID를 찾지 못한 경우
        requests.RequestException: 네트워크 오류
    """
    url = f"{_BASE}/compound/name/{drug_name}/cids/JSON"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise ValueError(f"'{drug_name}'에 대한 CID를 찾을 수 없습니다.")
        return int(cids[0])
    except requests.exceptions.Timeout:
        raise requests.RequestException(
            f"[오류] '{drug_name}' CID 조회 중 타임아웃 발생 ({timeout}초 초과)."
        )
    except requests.exceptions.HTTPError as e:
        raise requests.RequestException(
            f"[오류] '{drug_name}' CID 조회 HTTP 오류: {e.response.status_code} {e.response.reason}"
        )
    except requests.exceptions.RequestException as e:
        raise requests.RequestException(
            f"[오류] '{drug_name}' CID 조회 중 네트워크 오류: {e}"
        )


# =============================================================================
# 2단계: CID → 분자 기술자 조회
# =============================================================================

def fetch_descriptors(cid: int, timeout: int = 10) -> dict:
    """PubChem CID로 분자 기술자(분자량·logP·H결합 수 등)를 조회한다.

    Args:
        cid: PubChem CID
        timeout: HTTP 요청 타임아웃 (초)

    Returns:
        분자 기술자 딕셔너리
        키: mw_pubchem, xlogp, hbond_donor, hbond_acceptor, tpsa, smiles

    Raises:
        ValueError: 응답에서 기술자를 파싱하지 못한 경우
        requests.RequestException: 네트워크/HTTP 오류
    """
    props = (
        "MolecularWeight,XLogP,HBondDonorCount,"
        "HBondAcceptorCount,TPSA,CanonicalSMILES"
    )
    url = f"{_BASE}/compound/cid/{cid}/property/{props}/JSON"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        prop_list = data.get("PropertyTable", {}).get("Properties", [])
        if not prop_list:
            raise ValueError(
                f"CID {cid}에 대한 분자 기술자 응답이 비어 있습니다."
            )
        p = prop_list[0]
        return {
            "mw_pubchem":      float(p.get("MolecularWeight", float("nan"))),
            "xlogp":           float(p.get("XLogP", float("nan"))),
            "hbond_donor":     int(p.get("HBondDonorCount", -1)),
            "hbond_acceptor":  int(p.get("HBondAcceptorCount", -1)),
            "tpsa":            float(p.get("TPSA", float("nan"))),
            "smiles":          str(p.get("CanonicalSMILES", "")),
        }
    except requests.exceptions.Timeout:
        raise requests.RequestException(
            f"[오류] CID {cid} 기술자 조회 중 타임아웃 발생 ({timeout}초 초과)."
        )
    except requests.exceptions.HTTPError as e:
        raise requests.RequestException(
            f"[오류] CID {cid} 기술자 조회 HTTP 오류: {e.response.status_code} {e.response.reason}"
        )
    except requests.exceptions.RequestException as e:
        raise requests.RequestException(
            f"[오류] CID {cid} 기술자 조회 중 네트워크 오류: {e}"
        )


# =============================================================================
# 통합 수집 함수
# =============================================================================

_NAN_ROW = {
    "cid":             None,
    "mw_pubchem":      float("nan"),
    "xlogp":           float("nan"),
    "hbond_donor":     None,
    "hbond_acceptor":  None,
    "tpsa":            float("nan"),
    "smiles":          "",
}


def collect_all(drugs: list[str] | None = None) -> pd.DataFrame:
    """params.DRUGS에 등록된 약물의 PubChem 분자 기술자를 수집한다.

    네트워크 실패 시 해당 약물 행을 NaN으로 채우고 계속 진행한다.

    Args:
        drugs: 조회할 약물명 목록. None이면 params.DRUGS 전체 키 사용.

    Returns:
        컬럼 [drug, cid, mw_pubchem, xlogp, hbond_donor,
               hbond_acceptor, tpsa, smiles] 의 DataFrame
    """
    if drugs is None:
        drugs = list(DRUGS.keys())

    rows = []
    for drug in drugs:
        print(f"[수집 중] {drug} ...")
        row = {"drug": drug}

        # 1단계: CID 조회
        try:
            cid = fetch_cid(drug)
            row["cid"] = cid
            print(f"  CID 획득: {cid}")
        except Exception as e:
            print(f"  {e}")
            print(f"  → '{drug}' 행을 NaN으로 채웁니다.")
            row.update(_NAN_ROW)
            rows.append(row)
            time.sleep(0.25)
            continue

        time.sleep(0.25)  # PubChem rate limit 회피 (초당 최대 5회)

        # 2단계: 분자 기술자 조회
        try:
            descs = fetch_descriptors(cid)
            row.update(descs)
            print(f"  MW={descs['mw_pubchem']:.2f}, XLogP={descs['xlogp']}, "
                  f"TPSA={descs['tpsa']}")
        except Exception as e:
            print(f"  {e}")
            print(f"  → '{drug}' 기술자를 NaN으로 채웁니다.")
            tmp = dict(_NAN_ROW)
            tmp.pop("cid")          # CID는 이미 확보했으므로 유지
            row.update(tmp)

        rows.append(row)
        time.sleep(0.25)

    cols = ["drug", "cid", "mw_pubchem", "xlogp",
            "hbond_donor", "hbond_acceptor", "tpsa", "smiles"]
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# 메인 가드
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  PubChem 분자 기술자 수집 시작")
    print("=" * 55)

    df = collect_all()

    # CSV 저장
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[저장 완료] {_OUTPUT_CSV}")

    # 콘솔 출력
    print("\n[수집 결과]")
    print(df.to_string(index=False))

    # params 분자량 vs PubChem 분자량 비교
    print("\n[분자량 교차 검증]")
    all_ok = True
    for _, row in df.iterrows():
        drug = row["drug"]
        mw_param = DRUGS[drug]["molecular_weight"]
        mw_pub = row["mw_pubchem"]
        if pd.isna(mw_pub):
            print(f"  ⚠ {drug}: PubChem 값 없음 (수집 실패)")
            all_ok = False
            continue
        diff = abs(mw_pub - mw_param)
        status = "OK" if diff < 1.0 else "경고"
        flag = "" if diff < 1.0 else "  ← 1.0 g/mol 이상 차이!"
        print(f"  [{status}] {drug}: params={mw_param:.2f}, "
              f"PubChem={mw_pub:.2f}, 차이={diff:.3f} g/mol{flag}")
        if diff >= 1.0:
            all_ok = False

    if all_ok:
        print("\n모든 약물의 분자량이 params.py 값과 ±1.0 g/mol 이내입니다.")
    else:
        print("\n일부 약물에서 분자량 불일치 또는 수집 실패가 발생했습니다. "
              "위 경고를 확인하세요.")

    # 최종 통과 판정
    valid_rows = df.dropna(subset=["cid"])
    if len(valid_rows) == len(df):
        print("\nSTEP 1B 통과")
    else:
        missing = len(df) - len(valid_rows)
        print(f"\n[주의] {missing}개 약물의 CID 수집 실패 — 네트워크 연결을 확인하세요.")


"""
PubChem CID 참고 (2024년 기준):
  ibuprofen     CID 3672
  naproxen      CID 156391
  celecoxib     CID 2662
  acetaminophen CID 1983
"""

"""Monte Carlo 샘플링으로 가상 환자 500명을 생성하고 PBPK를 일괄 실행하는 모듈."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from params import DRUGS
from pbpk_model import simulate_pbpk

# ---------------------------------------------------------------------------
# tqdm 선택적 사용 (없으면 print 기반 진행 출력으로 폴백)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

_DATA_DIR = Path(__file__).parent / "data"

# =============================================================================
# 샘플링 상수
# =============================================================================

# eGFR 그룹: 이름 / 균등분포 범위 / 선택 확률
_EGFR_GROUP_NAMES  = ['normal', 'mild', 'moderate', 'severe']
_EGFR_GROUP_RANGES = {'normal': (90, 140), 'mild': (60, 89),
                      'moderate': (30, 59), 'severe': (15, 29)}
_EGFR_GROUP_PROBS  = np.array([0.60, 0.25, 0.10, 0.05])

# CYP2C9 유전형 및 한국인 기반 집단 빈도
_CYP2C9_OPTIONS = ['*1/*1', '*1/*2', '*1/*3', '*2/*3', '*3/*3']
_CYP2C9_PROBS   = np.array([0.55, 0.25, 0.10, 0.07, 0.03])

# 약물 목록 (params.DRUGS 순서 고정)
_DRUG_NAMES = list(DRUGS.keys())

# CSV 출력 컬럼 순서
_COLUMNS = [
    'patient_id', 'drug', 'dose_mg', 'body_weight', 'age',
    'egfr', 'egfr_group', 'cyp2c9_genotype',
    'Cmax_blood', 'Cmax_tissue', 'Tmax_blood', 'AUC_blood', 'label',
]


# =============================================================================
# 가상 환자 샘플링
# =============================================================================

def sample_patient(rng: np.random.Generator) -> dict:
    """Monte Carlo 샘플링으로 가상 환자 1명의 개인화 변수를 생성한다.

    Args:
        rng: numpy random Generator 인스턴스 (재현성 보장용)

    Returns:
        body_weight, egfr, egfr_group, cyp2c9_genotype, drug, dose_mg를
        담은 딕셔너리
    """
    # 체중: N(63, 11) 정규분포, [35, 130] kg 클리핑
    bw = float(np.clip(rng.normal(63.0, 11.0), 35.0, 130.0))

    # 나이: N(50, 18) 정규분포, [1, 95] 클리핑 후 정수 변환
    # 실제 NSAID 복용 인구의 연령 분포(중년 이상 다수) 반영
    age = int(np.clip(round(rng.normal(50.0, 18.0)), 1, 95))

    # eGFR: 그룹 배정 후 그룹 내 균등분포
    egfr_group = str(rng.choice(_EGFR_GROUP_NAMES, p=_EGFR_GROUP_PROBS))
    lo, hi = _EGFR_GROUP_RANGES[egfr_group]
    egfr = float(rng.uniform(lo, hi))   # 연속형 균등분포

    # CYP2C9 유전형
    genotype = str(rng.choice(_CYP2C9_OPTIONS, p=_CYP2C9_PROBS))

    # 약물: 4종 균등 무작위
    drug = str(rng.choice(_DRUG_NAMES))

    # 용량: 약물별 dose_range 내 균등분포
    dose_lo, dose_hi = DRUGS[drug]['dose_range']
    dose_mg = float(rng.uniform(dose_lo, dose_hi))

    return {
        'body_weight':     bw,
        'age':             age,
        'egfr':            egfr,
        'egfr_group':      egfr_group,
        'cyp2c9_genotype': genotype,
        'drug':            drug,
        'dose_mg':         dose_mg,
    }


# =============================================================================
# 독성 위험 라벨링
# =============================================================================

def label_patient(sim: dict, drug: str) -> str:
    """PBPK 시뮬레이션 결과로 독성 위험 라벨을 결정한다.

    라벨 3단계:
        'toxic'       : Cmax_blood > 독성 임계 (즉시 위험)
        'dose_adjust' : Cmax_blood > 용량조정 임계 또는 활막 IC50 미달
        'standard'    : 안전 범위 내, 유효 표적 농도 도달
        'error'       : PBPK solver 실패

    Args:
        sim: simulate_pbpk() 반환 딕셔너리
        drug: 약물 식별자

    Returns:
        라벨 문자열
    """
    if not sim['success']:
        return 'error'

    toxic_thr  = DRUGS[drug]['toxic_cmax_mg_per_L']
    adjust_thr = DRUGS[drug]['adjust_cmax_mg_per_L']
    ic50       = DRUGS[drug]['IC50_synovium_mg_per_L']

    cmax_b = sim['Cmax_blood']
    cmax_t = sim['Cmax_tissue']

    if cmax_b > toxic_thr:
        return 'toxic'
    if cmax_b > adjust_thr:
        return 'dose_adjust'
    if cmax_t < ic50:
        return 'dose_adjust'   # 표적 도달 부족 → 용량 재고
    return 'standard'


# =============================================================================
# 데이터셋 일괄 생성
# =============================================================================

def generate_dataset(
    n: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """가상 환자 n명의 PBPK 시뮬레이션 데이터셋을 생성한다.

    Args:
        n: 생성할 가상 환자 수 (기본 500)
        seed: 재현성을 위한 난수 시드 (기본 42)
        verbose: 진행 상황 출력 여부

    Returns:
        patient_id, drug, dose_mg, body_weight, egfr, egfr_group,
        cyp2c9_genotype, Cmax_blood, Cmax_tissue, Tmax_blood,
        AUC_blood, label 컬럼을 가진 DataFrame (n행)
    """
    rng = np.random.default_rng(seed)
    rows = []
    error_count = 0
    t_start = time.time()

    if verbose:
        print(f"가상 환자 {n}명 생성 시작 (seed={seed}) ...")

    iterator = range(n)
    if _HAS_TQDM and verbose:
        iterator = _tqdm(iterator, desc='PBPK 시뮬레이션', unit='명')

    for i in iterator:
        # 1. 환자 변수 샘플링
        patient = sample_patient(rng)
        drug = patient['drug']

        # 2. PBPK 시뮬레이션 (시간 배열은 저장하지 않음 — 메모리 효율)
        sim = simulate_pbpk(
            drug              = drug,
            dose_mg           = patient['dose_mg'],
            body_weight       = patient['body_weight'],
            egfr              = patient['egfr'],
            cyp2c9_genotype   = patient['cyp2c9_genotype'],
            age               = patient['age'],
            t_end_h           = 24.0,
            t_step_h          = 0.1,
        )

        # 3. 라벨 결정
        lbl = label_patient(sim, drug)
        if lbl == 'error':
            error_count += 1

        # 4. 행 데이터 구성 (시간 배열 제외)
        rows.append({
            'patient_id':       i + 1,
            'drug':             drug,
            'dose_mg':          round(patient['dose_mg'], 2),
            'body_weight':      round(patient['body_weight'], 2),
            'age':              patient['age'],
            'egfr':             round(patient['egfr'], 2),
            'egfr_group':       patient['egfr_group'],
            'cyp2c9_genotype':  patient['cyp2c9_genotype'],
            'Cmax_blood':       round(sim['Cmax_blood'],  4) if sim['success'] else np.nan,
            'Cmax_tissue':      round(sim['Cmax_tissue'], 4) if sim['success'] else np.nan,
            'Tmax_blood':       round(sim['Tmax_blood'],  2) if sim['success'] else np.nan,
            'AUC_blood':        round(sim['AUC_blood'],   4) if sim['success'] else np.nan,
            'label':            lbl,
        })

        # tqdm 없을 때 매 50명마다 진행 출력
        if verbose and not _HAS_TQDM and (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            pct = (i + 1) / n * 100
            print(f"  [{i+1:>4}/{n}] {pct:.0f}%  "
                  f"경과 {elapsed:.1f}s  오류 {error_count}건")

    df = pd.DataFrame(rows, columns=_COLUMNS)

    if verbose:
        elapsed = time.time() - t_start
        err_rate = error_count / n * 100
        print(f"\n완료: {n}명 / {elapsed:.1f}초 / 오류 {error_count}건 ({err_rate:.1f}%)")

    return df


# =============================================================================
# 메인 가드
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  가상 환자 데이터셋 생성 (Monte Carlo PBPK)")
    print("=" * 60)

    df = generate_dataset(n=500, seed=42, verbose=True)

    # age 컬럼 존재 확인
    assert 'age' in df.columns, "age 컬럼이 없음"

    # 자가 검증
    assert len(df) == 500, f"행 수 오류: {len(df)}"
    valid_labels = {'standard', 'dose_adjust', 'toxic', 'error'}
    assert set(df['label']).issubset(valid_labels), \
        f"알 수 없는 라벨: {set(df['label']) - valid_labels}"
    err_rate = (df['label'] == 'error').mean()
    assert err_rate < 0.05, f"오류 비율 {err_rate:.1%} ≥ 5%"
    assert df['drug'].nunique() == 4, \
        f"약물 종류 오류: {df['drug'].unique()}"
    for lbl in ('standard', 'dose_adjust', 'toxic'):
        assert (df['label'] == lbl).sum() >= 1, \
            f"'{lbl}' 라벨이 1건도 없음"

    # CSV 저장
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _DATA_DIR / 'patients_dataset.csv'
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\n[저장] {out_csv}  ({len(df)}행)")

    # 라벨 분포
    print("\n[라벨 분포]")
    print(df['label'].value_counts().to_string())

    # 약물 × 라벨 교차표
    print("\n[약물 × 라벨 교차표]")
    ct = pd.crosstab(df['drug'], df['label'])
    # 열 순서 정렬
    col_order = [c for c in ('standard', 'dose_adjust', 'toxic', 'error')
                 if c in ct.columns]
    print(ct[col_order].to_string())

    # 연령대별 분포
    print("\n[연령대별 분포]")
    age_bins   = [0, 19, 29, 39, 49, 59, 69, 95]
    age_labels = ['10대이하', '20대', '30대', '40대', '50대', '60대', '70대이상']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
    print(df['age_group'].value_counts().sort_index().to_string())
    df = df.drop(columns=['age_group'])   # 임시 컬럼 제거

    # eGFR 그룹 분포 확인
    print("\n[eGFR 그룹 분포]")
    print(df['egfr_group'].value_counts().to_string())

    # CYP2C9 분포 확인
    print("\n[CYP2C9 유전형 분포]")
    print(df['cyp2c9_genotype'].value_counts().to_string())

    # 오류 비율 경고
    err_pct = (df['label'] == 'error').mean() * 100
    if err_pct > 5:
        print(f"\n[경고] 오류 비율 {err_pct:.1f}% — 5% 초과. "
              "PBPK 파라미터 또는 solver 설정을 점검하세요.")
    else:
        print(f"\n오류 비율 {err_pct:.1f}% - 정상 범위 (< 5%)")

    print("\nSTEP 3 통과")

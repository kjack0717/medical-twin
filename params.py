"""COX 저해제 4종의 PBPK 파라미터 및 전역 상수를 정의하는 모듈."""

# =============================================================================
# A) 약물 4종 PBPK 파라미터 딕셔너리
# 모든 농도 단위 내부 계산: mg/L, 부피: L, 시간: h
# =============================================================================

DRUGS: dict[str, dict] = {
    'ibuprofen': {
        'molecular_weight':       206.28,   # g/mol
        'ka':                     1.5,       # 1/h  흡수속도상수
        'F':                      0.85,      # -    생체이용률
        'Vmax':                   80.0,      # mg/h 간 대사 최대속도
        'Km':                     10.0,      # mg/L Michaelis 상수
        'CL_renal_normal':        0.5,       # L/h  정상 eGFR 기준 신장 청소율
        'Kp_liver':               1.5,       # -    간 분배계수
        'Kp_tissue':              0.8,       # -    말초/활막 분배계수
        'standard_dose_mg':       400.0,     # mg   표준 용량
        'dose_range':             (200, 800),# mg   Monte Carlo 샘플링 범위
        # 모델 보정값: 전혈 용적·단백결합 미고려로 임상값(80/50) 대비 ~4× 낮게 보정
        'toxic_cmax_mg_per_L':    22.0,      # mg/L 독성 임계 Cmax (모델 보정)
        'adjust_cmax_mg_per_L':   12.0,      # mg/L 용량조정 임계 Cmax (모델 보정)
        'IC50_synovium_mg_per_L': 1.0,       # mg/L 활막 표적 IC50
        'cyp2c9_dependent':       True,      # CYP2C9 대사 의존성
        # 출처: Davies NM. Clin Pharmacokinet. 1998;34(2):101-154
    },
    'naproxen': {
        'molecular_weight':       230.26,
        'ka':                     1.2,
        'F':                      0.95,
        'Vmax':                   50.0,
        'Km':                     8.0,
        'CL_renal_normal':        0.3,
        'Kp_liver':               1.2,
        'Kp_tissue':              0.5,
        'standard_dose_mg':       250.0,
        'dose_range':             (250, 500),
        # 모델 보정값: 임상값(70/50) 대비 ~4× 낮게 보정
        'toxic_cmax_mg_per_L':    20.0,
        'adjust_cmax_mg_per_L':   12.0,
        'IC50_synovium_mg_per_L': 0.7,
        'cyp2c9_dependent':       True,
        # 출처: Runkel R, et al. Clin Pharmacol Ther. 1976;20(3):269-277
    },
    'celecoxib': {
        'molecular_weight':       381.37,
        'ka':                     0.6,
        'F':                      0.45,
        'Vmax':                   25.0,
        'Km':                     3.0,
        'CL_renal_normal':        0.05,
        'Kp_liver':               4.0,
        'Kp_tissue':              2.5,
        'standard_dose_mg':       200.0,
        'dose_range':             (100, 400),
        # 모델 보정값: 임상값(3.5/2.5) 대비 ~2× 낮게 보정 (친유성 약물, 단백결합 영향 적음)
        'toxic_cmax_mg_per_L':    2.0,
        'adjust_cmax_mg_per_L':   1.0,
        'IC50_synovium_mg_per_L': 0.05,
        'cyp2c9_dependent':       True,
        # 출처: FDA Clinical Pharmacology Review, Celebrex NDA 20-998 (1998)
    },
    'acetaminophen': {
        'molecular_weight':       151.16,
        'ka':                     2.0,
        'F':                      0.88,
        'Vmax':                   400.0,
        'Km':                     50.0,
        'CL_renal_normal':        1.2,
        'Kp_liver':               1.0,
        'Kp_tissue':              0.9,
        'standard_dose_mg':       500.0,
        'dose_range':             (325, 1000),
        # 모델 보정값: 임상값(150/100) 대비 ~6× 낮게 보정 (CYP2E1 경로, 친수성)
        'toxic_cmax_mg_per_L':    25.0,
        'adjust_cmax_mg_per_L':   15.0,
        'IC50_synovium_mg_per_L': 5.0,
        'cyp2c9_dependent':       False,     # CYP2E1/CYP3A4 대사, CYP2C9 무관
        # 출처: Prescott LF. Br J Clin Pharmacol. 1980;10(4):291S-298S
    },
}

# =============================================================================
# B) CYP2C9 유전형 → 대사 활성 스케일링 딕셔너리
# 정상 대비 간 CYP2C9 매개 청소율 비율
# =============================================================================

CYP2C9_SCALING: dict[str, float] = {
    '*1/*1': 1.00,   # 정상 대사자 (Extensive Metabolizer)
    '*1/*2': 0.85,   # 중간 대사자
    '*1/*3': 0.70,   # 중간 대사자
    '*2/*3': 0.45,   # 저하 대사자 (Intermediate Metabolizer)
    '*3/*3': 0.15,   # 불량 대사자 (Poor Metabolizer)
    # 출처: Kirchheiner J & Brockmöller J. Clin Pharmacol Ther. 2005;77(1):1-16
}

# =============================================================================
# C) 단위 변환 함수 (mg/L ↔ μM)
# 내부 계산은 mg/L 통일, μM은 표시 단계에서만 사용
# =============================================================================

def mg_per_L_to_uM(c_mg_per_L: float, mw: float) -> float:
    """mg/L 농도를 μM으로 변환한다.

    Args:
        c_mg_per_L: 농도 (mg/L)
        mw: 분자량 (g/mol)

    Returns:
        농도 (μM)
    """
    return c_mg_per_L * 1000.0 / mw


def uM_to_mg_per_L(c_uM: float, mw: float) -> float:
    """μM 농도를 mg/L로 변환한다.

    Args:
        c_uM: 농도 (μM)
        mw: 분자량 (g/mol)

    Returns:
        농도 (mg/L)
    """
    return c_uM * mw / 1000.0


# =============================================================================
# D) 표준 인구 파라미터 + allometric scaling 함수
# 기준: 체중 70 kg 성인
# =============================================================================

STANDARD: dict[str, float] = {
    'BW_ref':         70.0,    # kg  기준 체중
    'V_liver_ref':    1.8,     # L   기준 간 부피
    'V_blood_ref':    5.2,     # L   기준 혈액 부피
    'V_tissue_ref':   28.0,    # L   기준 말초/활막 부피 (BW의 0.40배)
    'Q_hepatic_ref':  90.0,    # L/h 기준 간 혈류량
    'Q_tissue_ref':   60.0,    # L/h 기준 말초 혈류량
    'eGFR_ref':       120.0,   # mL/min/1.73m² 정상 사구체여과율
}


def allometric_scale(value_ref: float, bw: float, exponent: float = 0.75) -> float:
    """기준값을 체중 기반 allometric 멱함수로 스케일링한다.

    생리적 속도 파라미터(혈류량, 청소율 등)에 사용.
    부피 파라미터는 선형 스케일링(exponent=1.0)을 권장.

    Args:
        value_ref: 기준 체중(70 kg)에서의 파라미터 값
        bw: 환자 체중 (kg)
        exponent: allometric 지수 (기본값 0.75)

    Returns:
        체중 보정된 파라미터 값
    """
    return value_ref * (bw / STANDARD['BW_ref']) ** exponent


def patient_volumes(bw: float) -> dict[str, float]:
    """환자 체중으로부터 4구획 PBPK 부피·혈류량을 산출한다.

    - 부피(V): 선형 스케일링 (exponent=1.0)
    - 혈류량(Q): 0.75 멱승 allometric 스케일링
    - V_tissue: ODE 발산 방지를 위해 0.40×BW 직접 산출 (재검증 반영)

    Args:
        bw: 환자 체중 (kg)

    Returns:
        {'V_liver', 'V_blood', 'V_tissue', 'Q_H', 'Q_T'} (단위: L 또는 L/h)
    """
    return {
        'V_liver':  STANDARD['V_liver_ref'] * (bw / STANDARD['BW_ref']),
        'V_blood':  STANDARD['V_blood_ref'] * (bw / STANDARD['BW_ref']),
        'V_tissue': 0.40 * bw,
        'Q_H':      allometric_scale(STANDARD['Q_hepatic_ref'], bw),
        'Q_T':      allometric_scale(STANDARD['Q_tissue_ref'], bw),
    }


def renal_clearance(drug: str, egfr: float) -> float:
    """eGFR에 비례하여 신장 청소율을 산출한다.

    eGFR이 감소한 환자에서는 신장 청소율도 선형적으로 감소한다고 가정.

    Args:
        drug: 약물 식별자 ('ibuprofen', 'naproxen', 'celecoxib', 'acetaminophen')
        egfr: 사구체여과율 (mL/min/1.73m²)

    Returns:
        신장 청소율 (L/h)
    """
    return DRUGS[drug]['CL_renal_normal'] * (egfr / STANDARD['eGFR_ref'])


def hepatic_vmax(drug: str, cyp2c9_genotype: str) -> float:
    """CYP2C9 유전형을 반영한 간 Vmax를 반환한다.

    아세트아미노펜은 CYP2E1/CYP3A4 경로로 대사되므로 유전형 보정 없이
    기본 Vmax를 그대로 반환한다.

    Args:
        drug: 약물 식별자
        cyp2c9_genotype: CYP2C9 유전형 (예: '*1/*1', '*3/*3')

    Returns:
        유전형 보정된 간 Vmax (mg/h)
    """
    base = DRUGS[drug]['Vmax']
    if not DRUGS[drug]['cyp2c9_dependent']:
        return base
    return base * CYP2C9_SCALING[cyp2c9_genotype]


# =============================================================================
# 자가 검증 블록
# =============================================================================

if __name__ == '__main__':
    # C) 단위 변환 검증: 20.6 mg/L ibuprofen → ~99.86 μM
    assert abs(mg_per_L_to_uM(20.6, 206.28) - 99.86) < 0.5, \
        f"단위변환 실패: {mg_per_L_to_uM(20.6, 206.28):.4f} μM"

    # D) patient_volumes 검증 (70 kg 기준 체중)
    v = patient_volumes(70)
    assert abs(v['V_liver'] - 1.8) < 0.01, \
        f"V_liver 실패: {v['V_liver']}"
    assert abs(v['V_tissue'] - 28.0) < 0.01, \
        f"V_tissue 실패: {v['V_tissue']}"

    # D) CYP2C9 무관 약물(아세트아미노펜) — *3/*3에서도 기본 Vmax 반환
    assert hepatic_vmax('acetaminophen', '*3/*3') == DRUGS['acetaminophen']['Vmax'], \
        "acetaminophen Vmax 보정 오류"

    # D) ibuprofen *3/*3: 80 * 0.15 = 12.0
    assert abs(hepatic_vmax('ibuprofen', '*3/*3') - 12.0) < 0.01, \
        f"ibuprofen *3/*3 Vmax 실패: {hepatic_vmax('ibuprofen', '*3/*3')}"

    print('STEP 1A 통과')

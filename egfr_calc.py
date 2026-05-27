"""CKD-EPI 2021 race-free eGFR 계산 모듈."""


def calculate_egfr_ckdepi(creatinine_mg_dl: float, age: int, sex: str) -> float:
    """CKD-EPI 2021 race-free 공식으로 eGFR(mL/min/1.73m²)을 계산한다.

    Args:
        creatinine_mg_dl: 혈청 크레아티닌 (mg/dL)
        age: 나이 (세)
        sex: 'female' 또는 'male'

    Returns:
        eGFR 값 (15.0 ~ 140.0 범위로 클립, 소수 첫째 자리 반올림)
    """
    # 성별에 따른 kappa(정규화 기준값), alpha(지수) 결정
    kappa = 0.7 if sex == 'female' else 0.9
    alpha = -0.241 if sex == 'female' else -0.302

    # Scr/kappa 비율 계산
    ratio = creatinine_mg_dl / kappa

    # CKD-EPI 2021 공식
    # min(ratio, 1)^alpha : Scr이 kappa 이하일 때의 기여
    # max(ratio, 1)^(-1.200) : Scr이 kappa 초과일 때의 기여
    # 0.9938^age : 나이에 따른 감소 보정
    # 1.012 (여성 보정 계수)
    egfr = (
        142
        * min(ratio, 1.0) ** alpha
        * max(ratio, 1.0) ** (-1.200)
        * (0.9938 ** age)
        * (1.012 if sex == 'female' else 1.0)
    )

    # 임상적으로 유효한 범위(15~140)로 클립 후 반올림
    return round(max(15.0, min(140.0, egfr)), 1)


if __name__ == '__main__':
    # 검증: 40세 남성 Cr=1.0 → ~99, 70세 여성 Cr=1.5 → ~38
    print(calculate_egfr_ckdepi(1.0, 40, 'male'))    # 예상: ~99
    print(calculate_egfr_ckdepi(1.5, 70, 'female'))  # 예상: ~38

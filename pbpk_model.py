"""PBPK ODE 시스템을 정의하고 scipy로 풀어 혈중·조직 농도를 시뮬레이션하는 모듈."""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# NumPy 2.0에서 trapz → trapezoid로 변경됨, 하위 호환 처리
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from params import (
    DRUGS,
    patient_volumes,
    renal_clearance,
    hepatic_vmax,
)

# ---------------------------------------------------------------------------
# 한국어 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, fallback: sans-serif)
# ---------------------------------------------------------------------------
def _setup_korean_font() -> None:
    """운영체제별 한국어 폰트를 자동으로 설정한다."""
    import platform
    candidates = {
        'Windows': 'Malgun Gothic',
        'Darwin':  'AppleGothic',
        'Linux':   'NanumGothic',
    }
    font = candidates.get(platform.system(), 'sans-serif')
    matplotlib.rcParams['font.family'] = font
    matplotlib.rcParams['axes.unicode_minus'] = False

_setup_korean_font()

_DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# ODE 우변 함수 (내부용)
# 상태 벡터: y = [A_gut(mg), C_liver(mg/L), C_blood(mg/L), C_tissue(mg/L)]
# 단위 검증:
#   dA_gut/dt    : (1/h)*(mg)          = mg/h        ✓
#   dC_liver/dt  : (L/h/L)*(mg/L)      = mg/(L·h)    ✓
#                  (mg/h) / (mg/L) 분자·분모 상쇄 후 /L = mg/(L·h) ✓
#   dC_blood/dt  : (mg)*(1/h)/L + ... = mg/(L·h)    ✓
#   dC_tissue/dt : (L/h/L)*(mg/L)      = mg/(L·h)    ✓
# =============================================================================

def _ode_system(t: float, y: np.ndarray, ctx: dict) -> list[float]:
    """4구획 PBPK ODE 우변. ctx 딕셔너리로 모든 파라미터를 전달받는다."""
    A_gut, C_liver, C_blood, C_tissue = y

    # 수치 오류로 인한 음수값 방지 (물리적으로 불가능한 상태)
    A_gut    = max(A_gut,    0.0)
    C_liver  = max(C_liver,  0.0)
    C_blood  = max(C_blood,  0.0)
    C_tissue = max(C_tissue, 0.0)

    ka         = ctx['ka']           # 1/h
    F          = ctx['F']            # -
    Vmax_eff   = ctx['Vmax_eff']     # mg/h
    Km         = ctx['Km']           # mg/L
    Kp_liver   = ctx['Kp_liver']     # -
    Kp_tissue  = ctx['Kp_tissue']    # -
    V_liver    = ctx['V_liver']      # L
    V_blood    = ctx['V_blood']      # L
    V_tissue   = ctx['V_tissue']     # L
    Q_H        = ctx['Q_H']          # L/h  간 혈류량
    Q_T        = ctx['Q_T']          # L/h  말초 혈류량
    CL_renal   = ctx['CL_renal']     # L/h  신장 청소율

    # Michaelis-Menten 간 대사량 (mg/h)
    metabolism = Vmax_eff * C_liver / (Km + C_liver)

    # 구획 간 농도 구동력 (mg/L)
    grad_liver  = C_blood - C_liver  / Kp_liver   # 혈액→간 구동력
    grad_tissue = C_blood - C_tissue / Kp_tissue  # 혈액→말초 구동력

    # --- ODE 우변 ---
    # 장 내 약물량 (mg/h)
    dA_gut_dt = -ka * A_gut

    # 간 농도 (mg/(L·h)): 혈류 유입/유출 + 대사 제거
    dC_liver_dt = (Q_H / V_liver) * grad_liver \
                  - metabolism / V_liver

    # 혈액 농도 (mg/(L·h)): 장 흡수 + 간교환 + 말초교환 + 신장배설
    dC_blood_dt = (ka * A_gut * F) / V_blood \
                  - (Q_H  / V_blood) * grad_liver  \
                  - (Q_T  / V_blood) * grad_tissue \
                  - (CL_renal * C_blood) / V_blood

    # 말초/활막 농도 (mg/(L·h))
    dC_tissue_dt = (Q_T / V_tissue) * grad_tissue

    return [dA_gut_dt, dC_liver_dt, dC_blood_dt, dC_tissue_dt]


# =============================================================================
# 핵심 시뮬레이션 함수
# =============================================================================

def simulate_pbpk(
    drug: str,
    dose_mg: float,
    body_weight: float,
    egfr: float,
    cyp2c9_genotype: str,
    t_end_h: float = 24.0,
    t_step_h: float = 0.1,
) -> dict:
    """4구획 PBPK ODE를 수치적으로 풀어 농도-시간 곡선과 PK 지표를 반환한다.

    Args:
        drug: 약물 식별자 ('ibuprofen', 'naproxen', 'celecoxib', 'acetaminophen')
        dose_mg: 투여 용량 (mg)
        body_weight: 환자 체중 (kg)
        egfr: 사구체여과율 (mL/min/1.73m²)
        cyp2c9_genotype: CYP2C9 유전형 (예: '*1/*1', '*3/*3')
        t_end_h: 시뮬레이션 종료 시간 (h), 기본 24시간
        t_step_h: 출력 시간 간격 (h), 기본 0.1시간

    Returns:
        dict:
            't'           (ndarray): 시간 배열 (h)
            'A_gut'       (ndarray): 장 약물량 (mg)
            'C_liver'     (ndarray): 간 농도 (mg/L)
            'C_blood'     (ndarray): 혈중 농도 (mg/L)
            'C_tissue'    (ndarray): 말초/활막 농도 (mg/L)
            'Cmax_blood'  (float):   혈중 최대 농도 (mg/L)
            'Cmax_tissue' (float):   활막 최대 농도 (mg/L)
            'Tmax_blood'  (float):   혈중 Cmax 도달 시간 (h)
            'AUC_blood'   (float):   혈중 AUC (mg·h/L, 사다리꼴 적분)
            'meta'        (dict):    입력 파라미터 echo
            'success'     (bool):    solver 성공 여부
    """
    dp   = DRUGS[drug]
    vols = patient_volumes(body_weight)
    cl_r = renal_clearance(drug, egfr)
    vmax = hepatic_vmax(drug, cyp2c9_genotype)

    ctx = {
        'ka':        dp['ka'],
        'F':         dp['F'],
        'Vmax_eff':  vmax,
        'Km':        dp['Km'],
        'Kp_liver':  dp['Kp_liver'],
        'Kp_tissue': dp['Kp_tissue'],
        'V_liver':   vols['V_liver'],
        'V_blood':   vols['V_blood'],
        'V_tissue':  vols['V_tissue'],
        'Q_H':       vols['Q_H'],
        'Q_T':       vols['Q_T'],
        'CL_renal':  cl_r,
    }

    t_eval = np.arange(0.0, t_end_h + t_step_h * 0.5, t_step_h)
    y0 = [dose_mg, 0.0, 0.0, 0.0]

    meta = {
        'drug':             drug,
        'dose_mg':          dose_mg,
        'body_weight':      body_weight,
        'egfr':             egfr,
        'cyp2c9_genotype':  cyp2c9_genotype,
        'Vmax_eff':         vmax,
        'CL_renal':         cl_r,
        'V_blood':          vols['V_blood'],
        'V_tissue':         vols['V_tissue'],
    }

    try:
        sol = solve_ivp(
            fun=lambda t, y: _ode_system(t, y, ctx),
            t_span=(0.0, t_end_h),
            y0=y0,
            method='LSODA',
            t_eval=t_eval,
            max_step=0.5,
            rtol=1e-6,
            atol=1e-9,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        t        = sol.t
        A_gut    = sol.y[0]
        C_liver  = sol.y[1]
        C_blood  = sol.y[2]
        C_tissue = sol.y[3]

        # PK 지표 계산
        Cmax_blood  = float(np.max(C_blood))
        Cmax_tissue = float(np.max(C_tissue))
        Tmax_blood  = float(t[np.argmax(C_blood)])
        AUC_blood   = float(_trapz(C_blood, t))  # 사다리꼴 적분

        return {
            't':          t,
            'A_gut':      A_gut,
            'C_liver':    C_liver,
            'C_blood':    C_blood,
            'C_tissue':   C_tissue,
            'Cmax_blood':  Cmax_blood,
            'Cmax_tissue': Cmax_tissue,
            'Tmax_blood':  Tmax_blood,
            'AUC_blood':   AUC_blood,
            'meta':        meta,
            'success':     True,
        }

    except Exception as exc:
        warnings.warn(f"[PBPK 오류] {drug} 시뮬레이션 실패: {exc}")
        nan_arr = np.full_like(t_eval, np.nan)
        meta['error'] = str(exc)
        return {
            't':          t_eval,
            'A_gut':      nan_arr.copy(),
            'C_liver':    nan_arr.copy(),
            'C_blood':    nan_arr.copy(),
            'C_tissue':   nan_arr.copy(),
            'Cmax_blood':  np.nan,
            'Cmax_tissue': np.nan,
            'Tmax_blood':  np.nan,
            'AUC_blood':   np.nan,
            'meta':        meta,
            'success':     False,
        }


# =============================================================================
# 시각화 함수
# =============================================================================

def plot_concentration_curves(
    sim_result: dict,
    save_path: str | None = None,
) -> None:
    """PBPK 시뮬레이션 결과를 2×2 서브플롯으로 시각화한다.

    Args:
        sim_result: simulate_pbpk()가 반환한 결과 딕셔너리
        save_path: 저장 경로 (None이면 plt.show() 호출)
    """
    meta  = sim_result['meta']
    drug  = meta['drug']
    dp    = DRUGS[drug]
    t     = sim_result['t']

    genotype_str = meta.get('cyp2c9_genotype', '?')
    title_info   = (
        f"{drug.capitalize()}  |  {meta['dose_mg']:.0f} mg  |  "
        f"체중 {meta['body_weight']:.0f} kg  |  "
        f"eGFR {meta['egfr']:.0f}  |  CYP2C9 {genotype_str}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    fig.suptitle(title_info, fontsize=11, fontweight='bold')

    # --- 서브플롯 1: 장 내 약물량 ---
    ax = axes[0, 0]
    ax.plot(t, sim_result['A_gut'], color='saddlebrown', lw=2)
    ax.set_title('장 내 약물량')
    ax.set_xlabel('시간 (h)')
    ax.set_ylabel('약물량 (mg)')
    ax.grid(True, alpha=0.3)

    # --- 서브플롯 2: 간 농도 ---
    ax = axes[0, 1]
    ax.plot(t, sim_result['C_liver'], color='darkorange', lw=2)
    ax.set_title('간 농도')
    ax.set_xlabel('시간 (h)')
    ax.set_ylabel('농도 (mg/L)')
    ax.grid(True, alpha=0.3)

    # --- 서브플롯 3: 혈중 농도 ---
    ax = axes[1, 0]
    ax.plot(t, sim_result['C_blood'], color='crimson', lw=2, label='혈중 농도')
    # 독성 임계선 (점선 적색)
    ax.axhline(dp['toxic_cmax_mg_per_L'],  color='red',    ls='--', lw=1.2,
               label=f"독성 임계 {dp['toxic_cmax_mg_per_L']} mg/L")
    # 용량조정 임계선 (점선 주황)
    ax.axhline(dp['adjust_cmax_mg_per_L'], color='orange', ls='--', lw=1.2,
               label=f"용량조정 임계 {dp['adjust_cmax_mg_per_L']} mg/L")
    # Cmax 표시
    ax.axhline(sim_result['Cmax_blood'], color='crimson', ls=':', lw=1.0,
               label=f"Cmax {sim_result['Cmax_blood']:.2f} mg/L")
    ax.set_title('혈중 농도')
    ax.set_xlabel('시간 (h)')
    ax.set_ylabel('농도 (mg/L)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- 서브플롯 4: 말초/활막 농도 ---
    ax = axes[1, 1]
    ax.plot(t, sim_result['C_tissue'], color='steelblue', lw=2, label='활막 농도')
    # IC50 표시 (점선 청색)
    ax.axhline(dp['IC50_synovium_mg_per_L'], color='blue', ls='--', lw=1.2,
               label=f"IC50 {dp['IC50_synovium_mg_per_L']} mg/L")
    # Cmax_tissue 표시
    ax.axhline(sim_result['Cmax_tissue'], color='steelblue', ls=':', lw=1.0,
               label=f"Cmax {sim_result['Cmax_tissue']:.2f} mg/L")
    ax.set_title('말초/활막 농도')
    ax.set_xlabel('시간 (h)')
    ax.set_ylabel('농도 (mg/L)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"  [저장] {out}")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# 자가 검증 블록
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  STEP 2 PBPK 모델 검증")
    print("=" * 55)

    # 케이스 1: 표준 성인, 정상 유전형, 표준 용량
    print("\n[케이스 1] 표준 성인 (70 kg / eGFR 120 / *1/*1)")
    r1 = simulate_pbpk('ibuprofen', 400, 70, 120, '*1/*1')
    assert r1['success'], f"표준 케이스 solver 실패: {r1['meta'].get('error')}"
    assert r1['Cmax_blood'] < 200, \
        f"표준 케이스 Cmax 비정상: {r1['Cmax_blood']:.4f} mg/L"
    print(f"  Cmax_blood  = {r1['Cmax_blood']:.4f} mg/L")
    print(f"  Tmax_blood  = {r1['Tmax_blood']:.2f} h")
    print(f"  AUC_blood   = {r1['AUC_blood']:.2f} mg·h/L")

    # 케이스 2: 저체중·신부전·*3/*3 → Cmax가 케이스 1보다 높아야 함
    print("\n[케이스 2] 취약 환자 (45 kg / eGFR 40 / *3/*3)")
    r2 = simulate_pbpk('ibuprofen', 400, 45, 40, '*3/*3')
    assert r2['success'], f"취약 케이스 solver 실패: {r2['meta'].get('error')}"
    assert r2['Cmax_blood'] > r1['Cmax_blood'], \
        (f"취약 케이스가 표준보다 낮음 — 모델 오류\n"
         f"  표준={r1['Cmax_blood']:.4f}, 취약={r2['Cmax_blood']:.4f}")
    print(f"  Cmax_blood  = {r2['Cmax_blood']:.4f} mg/L")
    print(f"  Tmax_blood  = {r2['Tmax_blood']:.2f} h")
    print(f"  AUC_blood   = {r2['AUC_blood']:.2f} mg·h/L")

    print(f"\n표준 Cmax_blood = {r1['Cmax_blood']:.2f}, "
          f"취약 Cmax_blood = {r2['Cmax_blood']:.2f}")

    # 그래프 저장
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    plot_concentration_curves(r1, save_path='data/pbpk_check_standard.png')
    plot_concentration_curves(r2, save_path='data/pbpk_check_vulnerable.png')

    print('\nSTEP 2 통과')

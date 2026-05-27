"""Streamlit 웹 앱 — 사용자 입력을 받아 개인 맞춤형 독성 위험을 시각화하는 모듈."""

# ── 페이지 설정은 반드시 최상단 (다른 st 호출보다 먼저) ──────────────────────
import streamlit as st

st.set_page_config(
    page_title='COX 메디컬 트윈',
    page_icon='💊',
    layout='wide',
)

# ── 나머지 임포트 ────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import joblib

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from params import DRUGS, CYP2C9_SCALING
from pbpk_model import simulate_pbpk

_MODEL_DIR = _ROOT / "models"

# CYP2C9 순서형 인코딩 (ml_model.py와 동일)
_CYP2C9_ORDER: dict[str, int] = {
    '*1/*1': 0, '*1/*2': 1, '*1/*3': 2, '*2/*3': 3, '*3/*3': 4,
}

# 라벨 한국어 설명
_LABEL_INFO: dict[str, dict] = {
    'standard': {
        'color': '#2E7D32', 'bg': '#E8F5E9',
        'badge': '안전 (표준)',
        'msg': '현재 투약 계획은 독성 위험 범위 안에 있습니다.',
    },
    'dose_adjust': {
        'color': '#E65100', 'bg': '#FFF3E0',
        'badge': '주의 (용량 조정)',
        'msg': '용량을 줄이거나 투약 간격을 늘리는 것을 고려하세요.',
    },
    'toxic': {
        'color': '#B71C1C', 'bg': '#FFEBEE',
        'badge': '위험 (독성 가능)',
        'msg': '이 조합은 독성 임계를 초과할 위험이 있습니다. 즉시 의사와 상담하세요.',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 캐싱: 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='모델 파일 로딩 중...')
def load_models() -> dict | None:
    """models/ 폴더에서 RF 모델과 전처리기를 로드한다.

    Returns:
        dict 또는 None (파일 누락 시)
    """
    required = {
        'rf':       'model_rf.pkl',
        'scaler':   'feature_scaler.pkl',
        'label_enc':'label_encoder.pkl',
        'drug_enc': 'drug_encoder.pkl',
    }
    missing = [v for v in required.values() if not (_MODEL_DIR / v).exists()]
    if missing:
        return None
    return {k: joblib.load(_MODEL_DIR / v) for k, v in required.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 캐싱: PBPK 시뮬레이션
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner='PBPK 시뮬레이션 실행 중...')
def cached_pbpk(
    drug: str, dose_mg: float, body_weight: float,
    egfr: float, cyp2c9_genotype: str,
) -> dict:
    """입력 조합이 동일하면 재계산을 생략하는 PBPK 래퍼."""
    return simulate_pbpk(drug, dose_mg, body_weight, egfr, cyp2c9_genotype)


# ─────────────────────────────────────────────────────────────────────────────
# 예측 전처리
# ─────────────────────────────────────────────────────────────────────────────

def predict_risk(
    models: dict,
    drug: str,
    dose_mg: float,
    body_weight: float,
    egfr: float,
    cyp2c9_genotype: str,
) -> tuple[str, np.ndarray]:
    """RF 모델로 독성 위험 라벨과 예측 확률을 반환한다.

    Returns:
        (label_str, proba_array)
    """
    drug_enc_val  = int(models['drug_enc'].transform([drug])[0])
    cyp2c9_enc_val = _CYP2C9_ORDER[cyp2c9_genotype]

    # 특성 순서: body_weight, egfr, cyp2c9_enc, drug_enc, dose_mg
    X_raw = np.array([[body_weight, egfr, cyp2c9_enc_val, drug_enc_val, dose_mg]])
    X_sc  = models['scaler'].transform(X_raw)

    pred_idx = models['rf'].predict(X_sc)[0]
    proba    = models['rf'].predict_proba(X_sc)[0]
    label    = models['label_enc'].inverse_transform([pred_idx])[0]
    return label, proba


# ─────────────────────────────────────────────────────────────────────────────
# UI 컴포넌트
# ─────────────────────────────────────────────────────────────────────────────

def render_risk_card(
    label: str,
    proba: np.ndarray,
    label_names: list[str],
) -> None:
    """위험 등급 배지·해설·확률 막대를 렌더링한다."""
    info = _LABEL_INFO.get(label, _LABEL_INFO['standard'])

    st.markdown(
        f"""
        <div style="
            background:{info['bg']};
            border-left: 6px solid {info['color']};
            border-radius: 8px;
            padding: 18px 24px;
            margin-bottom: 12px;
        ">
            <h2 style="color:{info['color']}; margin:0 0 6px 0;">
                예측 위험 등급: {info['badge']}
            </h2>
            <p style="font-size:1.05rem; margin:0; color:#333;">{info['msg']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 예측 확률 막대 (plotly)
    prob_colors = ['#2E7D32', '#E65100', '#B71C1C']  # standard, dose_adjust, toxic
    # label_enc는 알파벳 순: dose_adjust=0, standard=1, toxic=2
    name_map = {'dose_adjust': '주의(용량조정)', 'standard': '안전(표준)', 'toxic': '위험(독성)'}
    disp_names = [name_map.get(n, n) for n in label_names]
    # 색상을 label_names 순서에 맞게 재배열
    color_map = {'standard': '#2E7D32', 'dose_adjust': '#E65100', 'toxic': '#B71C1C'}
    bar_colors = [color_map.get(n, '#888') for n in label_names]

    fig_prob = go.Figure(go.Bar(
        x=disp_names,
        y=proba,
        marker_color=bar_colors,
        text=[f'{p:.1%}' for p in proba],
        textposition='outside',
    ))
    fig_prob.update_layout(
        title='클래스별 예측 확률',
        yaxis=dict(range=[0, 1.15], tickformat='.0%', title='확률'),
        xaxis_title='위험 등급',
        height=300,
        margin=dict(t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_prob, use_container_width=True)


def render_pk_curves(sim: dict, drug: str) -> None:
    """혈중·활막 농도-시간 곡선 및 PK 지표를 렌더링한다."""
    dp = DRUGS[drug]
    t  = sim['t']

    fig = go.Figure()

    # 혈중 농도
    fig.add_trace(go.Scatter(
        x=t, y=sim['C_blood'],
        mode='lines', name='혈중 농도',
        line=dict(color='crimson', width=2.5),
    ))
    # 활막 농도
    fig.add_trace(go.Scatter(
        x=t, y=sim['C_tissue'],
        mode='lines', name='활막/말초 농도',
        line=dict(color='steelblue', width=2.5),
    ))
    # 독성 임계선
    fig.add_hline(
        y=dp['toxic_cmax_mg_per_L'], line_dash='dash',
        line_color='red', line_width=1.5,
        annotation_text=f"독성 임계 {dp['toxic_cmax_mg_per_L']} mg/L",
        annotation_position='top right',
    )
    # 용량조정 임계선
    fig.add_hline(
        y=dp['adjust_cmax_mg_per_L'], line_dash='dot',
        line_color='orange', line_width=1.5,
        annotation_text=f"용량조정 임계 {dp['adjust_cmax_mg_per_L']} mg/L",
        annotation_position='top right',
    )
    # IC50
    fig.add_hline(
        y=dp['IC50_synovium_mg_per_L'], line_dash='longdash',
        line_color='steelblue', line_width=1.2,
        annotation_text=f"IC50 {dp['IC50_synovium_mg_per_L']} mg/L",
        annotation_position='bottom right',
    )

    fig.update_layout(
        title=f'{drug.capitalize()} 농도-시간 곡선 (24시간)',
        xaxis_title='시간 (h)',
        yaxis_title='농도 (mg/L)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420,
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # PK 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Cmax 혈중', f"{sim['Cmax_blood']:.3f} mg/L")
    c2.metric('Cmax 활막', f"{sim['Cmax_tissue']:.3f} mg/L")
    c3.metric('Tmax 혈중', f"{sim['Tmax_blood']:.2f} h")
    c4.metric('AUC₀₋₂₄ 혈중', f"{sim['AUC_blood']:.1f} mg·h/L")


def render_explainer() -> None:
    """메디컬 트윈 개념·모델 한계·면책 문구를 expander로 표시한다."""
    with st.expander('이 메디컬 트윈은 무엇을 하는가?', expanded=False):
        st.markdown("""
        ### 작동 원리
        1. **PBPK 모델** — 장→간→혈액→활막 4구획 약동학 ODE를 scipy로 실시간 풀어 약물 농도-시간 곡선을 생성합니다.
        2. **Monte Carlo 가상환자** — 체중·eGFR·CYP2C9 조합으로 500명의 다양한 환자를 시뮬레이션해 학습 데이터를 생성했습니다.
        3. **Random Forest 분류기** — 5-fold 교차검증 F1-macro 0.79를 달성한 모델이 독성 위험 등급을 예측합니다.

        ### 모델 한계
        - PBPK 모델은 단백결합(ibuprofen 99% 결합)과 활성 대사체를 명시적으로 모델링하지 않습니다.
        - 혈장 대신 전혈 부피를 사용하여 임상 농도를 과소 추정할 수 있습니다.
        - 학습 데이터는 실제 임상 데이터가 아닌 시뮬레이션 데이터입니다.
        - 약물 상호작용, 식사 영향, 제형 차이는 반영되지 않습니다.

        ### 참고 약물 정보
        | 약물 | 표준 용량 | CYP 경로 | IC50 (활막) |
        |------|----------|---------|------------|
        | 이부프로펜 | 400 mg | CYP2C9 | 1.0 mg/L |
        | 나프록센 | 250 mg | CYP2C9 | 0.7 mg/L |
        | 셀레콕시브 | 200 mg | CYP2C9 | 0.05 mg/L |
        | 아세트아미노펜 | 500 mg | CYP2E1 (CYP2C9 무관) | 5.0 mg/L |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 앱 레이아웃
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title('💊 COX 저해제 메디컬 트윈')
    st.caption('PBPK + Machine Learning 기반 개인 맞춤형 독성 위험 예측 시스템')

    # ── 모델 로드 확인 ─────────────────────────────────────────────────────
    models = load_models()
    if models is None:
        st.error(
            '모델 파일을 찾을 수 없습니다. '
            '`cd medical_twin && python ml_model.py` (STEP 4)를 먼저 실행하세요.'
        )
        st.stop()

    label_names: list[str] = list(models['label_enc'].classes_)

    # ── 사이드바 입력 ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header('환자 및 투약 정보 입력')

        drug = st.selectbox(
            '약물 선택',
            options=list(DRUGS.keys()),
            format_func=lambda x: {
                'ibuprofen': '이부프로펜 (Ibuprofen)',
                'naproxen': '나프록센 (Naproxen)',
                'celecoxib': '셀레콕시브 (Celecoxib)',
                'acetaminophen': '아세트아미노펜 (Acetaminophen)',
            }.get(x, x),
        )

        dp = DRUGS[drug]
        dose_lo, dose_hi = dp['dose_range']
        dose_mg = st.slider(
            '투여 용량 (mg)',
            min_value=int(dose_lo),
            max_value=int(dose_hi),
            value=int(dp['standard_dose_mg']),
            step=25,
        )

        body_weight = st.slider(
            '체중 (kg)',
            min_value=35, max_value=130, value=65, step=1,
        )

        egfr = st.slider(
            'eGFR (mL/min/1.73m²)',
            min_value=15, max_value=140, value=110, step=1,
            help='정상 ≥ 90 | 경도 60-89 | 중등도 30-59 | 중증 < 30',
        )

        cyp2c9_genotype = st.selectbox(
            'CYP2C9 유전형',
            options=list(CYP2C9_SCALING.keys()),
            index=0,
            help='*3/*3 = 불량 대사자(Poor Metabolizer) — 약물 농도 상승 위험',
        )

        st.divider()
        st.caption(
            f'선택 약물: **{drug}**  \n'
            f'용량조정 임계: {dp["adjust_cmax_mg_per_L"]} mg/L  \n'
            f'독성 임계: {dp["toxic_cmax_mg_per_L"]} mg/L'
        )

        if not dp['cyp2c9_dependent']:
            st.info('아세트아미노펜은 CYP2C9와 무관(CYP2E1 경로)하여 유전형 영향이 없습니다.')

    # ── PBPK 실행 ──────────────────────────────────────────────────────────
    sim = cached_pbpk(drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype)

    if not sim['success']:
        st.error(
            '현재 입력 조합에서 PBPK 시뮬레이션이 발산했습니다. '
            '체중·eGFR·용량 값을 변경한 후 다시 시도하세요.'
        )
        st.stop()

    # ── RF 예측 ────────────────────────────────────────────────────────────
    label, proba = predict_risk(
        models, drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype,
    )

    # ── 메인 패널 — 3섹션 ─────────────────────────────────────────────────
    # 섹션 1: 위험 등급 카드
    st.subheader('① 독성 위험 등급')
    render_risk_card(label, proba, label_names)

    st.divider()

    # 섹션 2: 농도-시간 곡선
    st.subheader('② 농도-시간 곡선 (PBPK 시뮬레이션)')
    render_pk_curves(sim, drug)

    st.divider()

    # 섹션 3: 설명 expander
    st.subheader('③ 모델 설명 및 주의사항')
    render_explainer()

    # ── 면책 문구 (하단 고정) ───────────────────────────────────────────────
    st.markdown('---')
    st.markdown(
        '<p style="text-align:center; color:#888; font-size:0.85rem;">'
        '⚠️ 본 도구는 교육·연구 목적의 시뮬레이터이며 실제 임상 결정에 사용해서는 안 됩니다. '
        '의약품 복용 전 반드시 의사 또는 약사와 상담하세요.'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()

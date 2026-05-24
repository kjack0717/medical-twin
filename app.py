"""나의 약물 메디컬 트윈 — Streamlit 웹 앱 v2."""

import streamlit as st

st.set_page_config(
    page_title='나의 약물 메디컬 트윈',
    page_icon='💊',
    layout='wide',
)

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

# ── 용어 한국어화 딕셔너리 ────────────────────────────────────────────────────

GLOSSARY = {
    'Cmax': '최고 혈중 농도',
    'Tmax': '최고 농도 도달 시간',
    'AUC': '총 약물 노출량',
    'PBPK': '생리기반 약동학 모델',
    'eGFR': '신장 기능 지표',
    'CYP2C9': '약물 분해 효소 유형',
    'IC50': '절반 억제 농도',
    'mg/L': 'mg/L (혈중 약물 농도 단위)',
    'toxic': '독성 위험',
    'dose_adjust': '용량 조정 필요',
    'standard': '정상 범위',
}

DRUG_KOREAN = {
    'ibuprofen':     '이부프로펜',
    'naproxen':      '나프록센',
    'celecoxib':     '셀레콕시브',
    'acetaminophen': '아세트아미노펜',
}

CYP2C9_KOREAN = {
    '*1/*1': '*1/*1 — 정상 분해 (약물이 빠르게 제거됨)',
    '*1/*2': '*1/*2 — 약간 느린 분해',
    '*1/*3': '*1/*3 — 느린 분해',
    '*2/*3': '*2/*3 — 매우 느린 분해 (약물 축적 주의)',
    '*3/*3': '*3/*3 — 거의 분해 안 됨 (가장 높은 축적 위험)',
}

LABEL_KOREAN = {
    'standard':    ('안전 — 정상 범위',     '#2E7D32', '#E8F5E9',
                    '현재 복용량은 안전 범위 안에 있습니다.'),
    'dose_adjust': ('주의 — 용량 줄이기 권장', '#E65100', '#FFF3E0',
                    '혈중 농도가 높아질 수 있습니다. 복용량을 줄이거나 복용 간격을 늘리세요.'),
    'toxic':       ('위험 — 독성 가능성',    '#B71C1C', '#FFEBEE',
                    '이 조합은 독성 수준에 도달할 위험이 있습니다. 즉시 의사와 상담하세요.'),
}

EGFR_GROUP_LABEL = {
    'normal':   '정상 (90 이상)',
    'mild':     '경도 저하 (60-89)',
    'moderate': '중등도 저하 (30-59)',
    'severe':   '중증 저하 (15-29)',
}

_CYP2C9_ORDER = {'*1/*1': 0, '*1/*2': 1, '*1/*3': 2, '*2/*3': 3, '*3/*3': 4}

# ── 모델 로드 ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='모델 파일 불러오는 중...')
def load_models():
    required = {
        'rf':        'model_rf.pkl',
        'scaler':    'feature_scaler.pkl',
        'label_enc': 'label_encoder.pkl',
        'drug_enc':  'drug_encoder.pkl',
    }
    missing = [v for v in required.values() if not (_MODEL_DIR / v).exists()]
    if missing:
        return None
    return {k: joblib.load(_MODEL_DIR / v) for k, v in required.items()}


# ── PBPK 시뮬레이션 ──────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner='약물 이동 경로 계산 중...')
def run_simulation(drug, dose_mg, body_weight, egfr, cyp2c9_genotype):
    return simulate_pbpk(drug, dose_mg, body_weight, egfr, cyp2c9_genotype)


# ── 장기별 위험도 계산 ────────────────────────────────────────────────────────

def compute_risk_organs(sim, drug):
    """각 장기(혈액·간·활막)의 위험 상태를 반환한다.

    Returns dict: organ -> 'safe' | 'warn' | 'danger'
    """
    dp = DRUGS[drug]
    toxic  = dp['toxic_cmax_mg_per_L']
    adjust = dp['adjust_cmax_mg_per_L']

    def _state(cmax):
        ratio = cmax / toxic
        if ratio >= 0.80:
            return 'danger'
        if ratio >= 0.50:
            return 'warn'
        return 'safe'

    return {
        '혈액': _state(sim['Cmax_blood']),
        '간':   _state(max(sim['C_liver'])),
        '활막': _state(sim['Cmax_tissue']),
    }


# ── Plotly 인체 그림 (애니메이션) ─────────────────────────────────────────────

_ORGAN_POS = {
    '장':  (-0.05, 0.38),
    '간':  ( 0.15, 0.52),
    '혈액':( 0.00, 0.60),
    '활막':( 0.20, 0.08),
}

_ORGAN_COLOR = {
    'safe':   '#00C853',
    'warn':   '#FFD600',
    'danger': '#D50000',
}

_COMPARTMENT_KEYS = {
    '장':   'A_gut',
    '간':   'C_liver',
    '혈액': 'C_blood',
    '활막': 'C_tissue',
}


def _organ_value(sim, organ, idx):
    key = _COMPARTMENT_KEYS[organ]
    arr = sim[key]
    v = arr[idx] if idx < len(arr) else arr[-1]
    return float(v)


def _organ_state_at(sim, drug, organ, idx):
    dp = DRUGS[drug]
    toxic = dp['toxic_cmax_mg_per_L']
    v = _organ_value(sim, organ, idx)
    if organ == '장':
        ratio = v / (dp['dose_range'][1] * dp['F'])
        ratio = min(ratio, 1.0)
    else:
        ratio = v / toxic
    if ratio >= 0.80:
        return 'danger'
    if ratio >= 0.50:
        return 'warn'
    return 'safe'


def generate_frames(sim, drug, step=5):
    n_pts = len(sim['t'])
    indices = list(range(0, n_pts, step))
    frames = []
    for i in indices:
        xs, ys, texts, colors, sizes = [], [], [], [], []
        for organ, (ox, oy) in _ORGAN_POS.items():
            state = _organ_state_at(sim, drug, organ, i)
            v = _organ_value(sim, organ, i)
            xs.append(ox)
            ys.append(oy)
            texts.append(f"{organ}<br>{v:.2f}")
            colors.append(_ORGAN_COLOR[state])
            sizes.append(38)
        frames.append(go.Frame(
            data=[go.Scatter(
                x=xs, y=ys,
                mode='markers+text',
                marker=dict(size=sizes, color=colors, opacity=0.9,
                            line=dict(width=2, color='white')),
                text=texts,
                textposition='top center',
            )],
            name=str(i),
        ))
    return frames, indices


def build_human_figure(sim, drug):
    frames, indices = generate_frames(sim, drug, step=5)
    n_pts = len(sim['t'])
    t_arr = sim['t']

    # 초기 프레임 (t=0)
    init_xs, init_ys, init_texts, init_colors = [], [], [], []
    for organ, (ox, oy) in _ORGAN_POS.items():
        state = _organ_state_at(sim, drug, organ, 0)
        v = _organ_value(sim, organ, 0)
        init_xs.append(ox)
        init_ys.append(oy)
        init_texts.append(f"{organ}<br>{v:.2f}")
        init_colors.append(_ORGAN_COLOR[state])

    # 인체 실루엣 (단순 타원형 윤곽)
    theta = np.linspace(0, 2 * np.pi, 80)
    body_x = 0.22 * np.cos(theta)
    body_y = 0.40 * np.sin(theta) + 0.40

    fig = go.Figure(
        data=[
            go.Scatter(
                x=body_x, y=body_y,
                mode='lines',
                line=dict(color='#90CAF9', width=2),
                fill='toself',
                fillcolor='rgba(144,202,249,0.10)',
                hoverinfo='skip',
                name='인체',
            ),
            go.Scatter(
                x=init_xs, y=init_ys,
                mode='markers+text',
                marker=dict(size=38, color=init_colors, opacity=0.9,
                            line=dict(width=2, color='white')),
                text=init_texts,
                textposition='top center',
                name='장기',
            ),
        ],
        frames=frames,
    )

    step_labels = [f"{t_arr[i]:.1f}h" for i in range(0, n_pts, 5)]
    slider_steps = [
        dict(
            args=[[str(i)], dict(frame=dict(duration=80, redraw=True), mode='immediate')],
            label=step_labels[k],
            method='animate',
        )
        for k, i in enumerate(range(0, n_pts, 5))
    ]

    fig.update_layout(
        title='장기별 약물 농도 (mg/L) — 시간에 따른 변화',
        xaxis=dict(range=[-0.45, 0.55], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.05, 1.05], showticklabels=False, showgrid=False, zeroline=False),
        height=480,
        paper_bgcolor='rgba(10,10,20,1)',
        plot_bgcolor='rgba(10,10,20,1)',
        font=dict(color='white'),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0.02, x=0.18,
            xanchor='right',
            buttons=[
                dict(label='▶ 재생', method='animate',
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ 정지', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.18, y=0,
            len=0.82,
            currentvalue=dict(prefix='시간: ', font=dict(color='white')),
            pad=dict(t=30),
        )],
    )
    return fig


# ── HTML5 Canvas 약물 경로 애니메이션 ──────────────────────────────────────────

def build_pathway_animation_html():
    """standalone HTML 파일을 그대로 읽어 반환한다."""
    html_path = _ROOT / "Medical Twin - Drug Pathway (Standalone).html"
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return "<p style='color:red;'>애니메이션 파일(Medical Twin - Drug Pathway (Standalone).html)을 찾을 수 없습니다.</p>"


# ── 안전 배너 ────────────────────────────────────────────────────────────────

def render_safety_banner(label):
    badge, color, bg, msg = LABEL_KOREAN.get(label, LABEL_KOREAN['standard'])
    st.markdown(
        f"""
        <div style="background:{bg};border-left:6px solid {color};
             border-radius:8px;padding:18px 24px;margin-bottom:14px;">
          <h2 style="color:{color};margin:0 0 6px 0;">예측 결과: {badge}</h2>
          <p style="font-size:1.05rem;margin:0;color:#333;">{msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 예측 ─────────────────────────────────────────────────────────────────────

def predict_risk(models, drug, dose_mg, body_weight, egfr, cyp2c9_genotype):
    drug_enc_val   = int(models['drug_enc'].transform([drug])[0])
    cyp2c9_enc_val = _CYP2C9_ORDER[cyp2c9_genotype]
    X_raw = np.array([[body_weight, egfr, cyp2c9_enc_val, drug_enc_val, dose_mg]])
    X_sc  = models['scaler'].transform(X_raw)
    pred_idx = models['rf'].predict(X_sc)[0]
    proba    = models['rf'].predict_proba(X_sc)[0]
    label    = models['label_enc'].inverse_transform([pred_idx])[0]
    return label, proba


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    st.title('💊 나의 약물 메디컬 트윈')
    st.caption('생리기반 약동학 모델(PBPK) + 머신러닝으로 나에게 맞는 약물 위험도를 예측합니다')

    models = load_models()
    if models is None:
        st.error('모델 파일을 찾을 수 없습니다. `python ml_model.py`를 먼저 실행해 주세요.')
        st.stop()

    label_names = list(models['label_enc'].classes_)

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('내 정보 입력')

        drug = st.selectbox(
            '복용할 약물',
            options=list(DRUGS.keys()),
            format_func=lambda x: f"{DRUG_KOREAN[x]} ({x.capitalize()})",
        )
        dp = DRUGS[drug]

        dose_lo, dose_hi = dp['dose_range']
        dose_mg = st.slider(
            '복용량 (mg)',
            min_value=int(dose_lo),
            max_value=int(dose_hi),
            value=int(dp['standard_dose_mg']),
            step=25,
        )

        body_weight = st.slider('체중 (kg)', 35, 130, 65, 1)

        egfr = st.slider(
            '신장 기능 지표 (eGFR, mL/min/1.73m²)',
            min_value=15, max_value=140, value=110, step=1,
            help='정상 90 이상 | 경도 저하 60-89 | 중등도 저하 30-59 | 중증 저하 15-29',
        )

        cyp2c9_genotype = st.selectbox(
            '약물 분해 효소 유형 (CYP2C9)',
            options=list(CYP2C9_SCALING.keys()),
            index=0,
            format_func=lambda x: CYP2C9_KOREAN[x],
        )

        st.divider()
        st.caption(
            f'**{DRUG_KOREAN[drug]}**  \n'
            f'주의 기준 농도: {dp["adjust_cmax_mg_per_L"]} mg/L  \n'
            f'위험 기준 농도: {dp["toxic_cmax_mg_per_L"]} mg/L'
        )
        if not dp['cyp2c9_dependent']:
            st.info('아세트아미노펜은 약물 분해 효소 유형(CYP2C9)의 영향을 받지 않습니다.')

    # ── 시뮬레이션 & 예측 ────────────────────────────────────────────────────
    sim = run_simulation(drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype)

    if not sim['success']:
        st.error('계산 오류가 발생했습니다. 체중·신장 기능·복용량 값을 바꿔 다시 시도해 주세요.')
        st.stop()

    label, proba = predict_risk(models, drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype)
    organ_states = compute_risk_organs(sim, drug)

    # ────────────────────────────────────────────────────────────────────────
    # 영역 1 — 위험 등급 배너
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('① 나의 위험 등급')
    render_safety_banner(label)

    # 예측 확률 막대
    name_map   = {'dose_adjust': '주의\n(용량 조정)', 'standard': '안전\n(정상)', 'toxic': '위험\n(독성)'}
    color_map  = {'standard': '#2E7D32', 'dose_adjust': '#E65100', 'toxic': '#B71C1C'}
    disp_names = [name_map.get(n, n) for n in label_names]
    bar_colors = [color_map.get(n, '#888') for n in label_names]

    fig_prob = go.Figure(go.Bar(
        x=disp_names, y=proba,
        marker_color=bar_colors,
        text=[f'{p:.1%}' for p in proba],
        textposition='outside',
    ))
    fig_prob.update_layout(
        title='등급별 예측 확률',
        yaxis=dict(range=[0, 1.2], tickformat='.0%', title='확률'),
        xaxis_title='위험 등급',
        height=280,
        margin=dict(t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 2 — 인체 장기 애니메이션
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('② 약물이 내 몸에서 이동하는 경로')

    # 장기 상태 요약 뱃지
    badge_cols = st.columns(len(organ_states))
    state_color = {'safe': '#2E7D32', 'warn': '#E65100', 'danger': '#B71C1C'}
    state_label = {'safe': '안전', 'warn': '주의', 'danger': '위험'}
    for col, (organ, state) in zip(badge_cols, organ_states.items()):
        col.markdown(
            f"<div style='text-align:center;background:{state_color[state]}20;"
            f"border:2px solid {state_color[state]};border-radius:8px;padding:8px;'>"
            f"<b style='color:{state_color[state]};font-size:1.1rem;'>{organ}</b><br>"
            f"<span style='color:{state_color[state]};'>{state_label[state]}</span></div>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(build_human_figure(sim, drug), use_container_width=True)

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 3 — 혈중 농도-시간 곡선
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('③ 시간에 따른 혈중 농도 변화')

    t  = sim['t']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sim['C_blood'],  mode='lines', name='혈중 농도',
                             line=dict(color='crimson', width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=sim['C_tissue'], mode='lines', name='활막/말초 농도',
                             line=dict(color='steelblue', width=2.5)))
    fig.add_hline(y=dp['toxic_cmax_mg_per_L'],  line_dash='dash',  line_color='red',
                  annotation_text=f"독성 기준 {dp['toxic_cmax_mg_per_L']} mg/L",
                  annotation_position='top right')
    fig.add_hline(y=dp['adjust_cmax_mg_per_L'], line_dash='dot',   line_color='orange',
                  annotation_text=f"주의 기준 {dp['adjust_cmax_mg_per_L']} mg/L",
                  annotation_position='top right')
    fig.add_hline(y=dp['IC50_synovium_mg_per_L'], line_dash='longdash', line_color='steelblue',
                  annotation_text=f"절반 억제 농도 {dp['IC50_synovium_mg_per_L']} mg/L",
                  annotation_position='bottom right')
    fig.update_layout(
        title=f'{DRUG_KOREAN[drug]} 농도-시간 곡선 (24시간)',
        xaxis_title='시간 (h)',
        yaxis_title='농도 (mg/L)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        plot_bgcolor='rgba(250,250,250,1)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('최고 혈중 농도 (Cmax)',   f"{sim['Cmax_blood']:.3f} mg/L")
    c2.metric('최고 활막 농도',          f"{sim['Cmax_tissue']:.3f} mg/L")
    c3.metric('최고 농도 도달 시간 (Tmax)', f"{sim['Tmax_blood']:.2f} h")
    c4.metric('총 약물 노출량 (AUC₀₋₂₄)', f"{sim['AUC_blood']:.1f} mg·h/L")

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 4 — 다른 복용량과 비교
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('④ 다른 복용량과 비교')

    compare_doses = [int(dose_lo), int(dp['standard_dose_mg']), int(dose_hi)]
    if dose_mg not in compare_doses:
        compare_doses.append(dose_mg)
    compare_doses = sorted(set(compare_doses))

    cmp_rows = []
    for d in compare_doses:
        s = run_simulation(drug, float(d), float(body_weight), float(egfr), cyp2c9_genotype)
        if s['success']:
            lbl, _ = predict_risk(models, drug, float(d), float(body_weight), float(egfr), cyp2c9_genotype)
            badge_txt, col_hex, _, _ = LABEL_KOREAN.get(lbl, LABEL_KOREAN['standard'])
            marker = ' ← 현재' if d == dose_mg else ''
            cmp_rows.append({
                '복용량 (mg)': f'{d}{marker}',
                '최고 혈중 농도': f'{s["Cmax_blood"]:.3f} mg/L',
                '총 약물 노출량': f'{s["AUC_blood"]:.1f} mg·h/L',
                '위험 등급': badge_txt,
            })

    if cmp_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 5 — 약물 경로 애니메이션 (Canvas)
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('⑤ 약이 몸속을 이동하는 모습 (실시간 애니메이션)')
    st.components.v1.html(
        build_pathway_animation_html(),
        height=700,
    )

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 6 — 모델 설명
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('⑥ 이 도구는 어떻게 작동하나요?')
    with st.expander('자세히 보기', expanded=False):
        st.markdown(f"""
### 작동 원리
1. **생리기반 약동학 모델 (PBPK)** — 약물이 장→간→혈액→활막으로 이동하는 과정을 수학 방정식으로 실시간 계산합니다.
2. **가상환자 500명 시뮬레이션** — 다양한 체중·신장 기능·효소 유형을 가진 가상 환자를 만들어 학습 데이터를 생성했습니다.
3. **머신러닝 분류기 (Random Forest)** — 5-겹 교차검증 F1-macro 0.79를 달성한 모델이 위험 등급을 예측합니다.

### 용어 설명
| 용어 | 설명 |
|------|------|
| 최고 혈중 농도 (Cmax) | 복용 후 혈액 속 약물 농도의 최대값 |
| 총 약물 노출량 (AUC) | 24시간 동안 혈액이 약물에 노출된 총량 |
| 신장 기능 지표 (eGFR) | 신장이 1분에 혈액을 얼마나 걸러내는지의 속도 |
| 약물 분해 효소 유형 (CYP2C9) | 간에서 이 약물을 분해하는 효소의 종류 — 유전자에 따라 다름 |
| 절반 억제 농도 (IC50) | 약이 염증을 절반으로 줄이는 데 필요한 최소 농도 |

### 약물 정보
| 약물 | 표준 복용량 | 효소 경로 | 절반 억제 농도 |
|------|------------|----------|---------------|
| 이부프로펜 | 400 mg | CYP2C9 | 1.0 mg/L |
| 나프록센 | 250 mg | CYP2C9 | 0.7 mg/L |
| 셀레콕시브 | 200 mg | CYP2C9 | 0.05 mg/L |
| 아세트아미노펜 | 500 mg | CYP2E1 (CYP2C9 무관) | 5.0 mg/L |

### 모델 한계
- 단백결합(이부프로펜 99%)과 활성 대사체를 명시적으로 반영하지 않습니다.
- 학습 데이터는 실제 임상 데이터가 아닌 시뮬레이션 데이터입니다.
- 약물 상호작용, 식사 영향, 제형 차이는 반영되지 않습니다.
        """)

    # ── 면책 문구 ─────────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown(
        '<p style="text-align:center;color:#888;font-size:0.85rem;">'
        '⚠️ 본 도구는 고등학교 연구 목적의 교육용 시뮬레이터입니다. '
        '실제 임상 결정에 사용해서는 안 됩니다. '
        '의약품 복용 전 반드시 의사 또는 약사와 상담하세요.'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()

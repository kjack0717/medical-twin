"""나의 약물 메디컬 트윈 — Streamlit 웹 앱 v2."""

import streamlit as st

st.set_page_config(
    page_title='나의 약물 메디컬 트윈',
    page_icon='💊',
    layout='wide',
)

import json
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import joblib

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from params import DRUGS, CYP2C9_SCALING
from pbpk_model import simulate_pbpk

_MODEL_DIR  = _ROOT / "models"
_ANIM_PATH  = _ROOT / "Medical Twin - Drug Pathway (Standalone).html"
_HTML_BASE  = None   # module-level HTML cache (read once per process)

# ── 용어 한국어화 딕셔너리 ────────────────────────────────────────────────────

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
    'standard':    ('안전 — 정상 범위',      '#2E7D32', '#E8F5E9',
                    '현재 복용량은 안전 범위 안에 있습니다.'),
    'dose_adjust': ('주의 — 용량 줄이기 권장', '#E65100', '#FFF3E0',
                    '혈중 농도가 높아질 수 있습니다. 복용량을 줄이거나 복용 간격을 늘리세요.'),
    'toxic':       ('위험 — 독성 가능성',     '#B71C1C', '#FFEBEE',
                    '이 조합은 독성 수준에 도달할 위험이 있습니다. 즉시 의사와 상담하세요.'),
}

_CYP2C9_ORDER = {'*1/*1': 0, '*1/*2': 1, '*1/*3': 2, '*2/*3': 3, '*3/*3': 4}

# ── Babel 오버라이드 스크립트 (모듈 상수) ────────────────────────────────────
# 번들러 template에 주입된다. 원본 JS globals(OrganPanel/getCurrentFocus 등)을
# 환자 위험 데이터 기반으로 덮어쓴다.

_PATHWAY_OVERRIDE_BABEL = r"""<script type="text/babel">
(function() {
  var __RISK = window.__patientRisk || {};

  /* 1. getCurrentFocus: 위험 장기만 focus zoom */
  var _origGCF = window.getCurrentFocus;
  window.getCurrentFocus = function(t) {
    var r = _origGCF(t);
    if (r.key && !__RISK[r.key]) return { key: null, p: 0 };
    return r;
  };

  /* 2. OrganPanel: 안전 장기 → 심플 패널(spread/zoom 없음) */
  var _origOP = window.OrganPanel;
  window.OrganPanel = function(props) {
    var cfg = props.cfg, t = props.t, activeAt = props.activeAt;
    var focusP = props.focusP, dimOpacity = props.dimOpacity;
    var organKey = Object.keys(PANELS).find(function(k) { return PANELS[k] === cfg; });
    var atRisk = organKey ? !!__RISK[organKey] : true;

    if (atRisk) return _origOP(props);

    /* 안전 장기 */
    var since = t - activeAt;
    var receiving = since >= 0 && since < SUB.delivery.e;
    var passed    = since >= SUB.delivery.e;

    return (
      <div style={{
        position: 'absolute', left: cfg.x, top: cfg.y,
        width: PANEL_W, height: PANEL_H,
        opacity: 1 - (dimOpacity || 0), zIndex: 1,
        background: C.boxBg,
        border: '3px solid ' + (passed ? '#2d5a2d' : C.boxStroke),
        borderRadius: 4, overflow: 'hidden',
        boxShadow: passed
          ? '0 0 0 2px #2d5a2d22, 0 4px 16px rgba(20,80,20,0.10)'
          : '0 4px 12px rgba(80,40,20,0.06)',
        transition: 'border-color 500ms, box-shadow 500ms',
      }}>
        <div style={{
          position: 'absolute', left: 50, top: 40,
          width: PANEL_W - 100, height: PANEL_H - 140,
          mixBlendMode: 'multiply',
        }}>
          <img src={cfg.src} alt=""
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
        </div>
        <div style={{
          position: 'absolute', left: 0, right: 0, bottom: 22, textAlign: 'center',
          fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif',
        }}>
          <div style={{ fontSize: 34, fontWeight: 700, color: C.inkSoft, letterSpacing: '-0.02em' }}>
            {cfg.label}
          </div>
          <div style={{ fontSize: 12, fontWeight: 500, color: C.inkSoft, opacity: 0.6,
            letterSpacing: '0.18em', textTransform: 'uppercase', marginTop: 2 }}>
            {cfg.sub}
          </div>
        </div>
        {since >= 0 && (
          <div style={{
            position: 'absolute', top: 12, right: 12, padding: '3px 9px',
            fontSize: 10, letterSpacing: '0.18em',
            fontFamily: 'JetBrains Mono, ui-monospace, monospace',
            textTransform: 'uppercase',
            background: passed ? '#2d5a2d' : 'transparent',
            color:      passed ? '#A5D6A7' : C.boxStroke,
            border: '1px solid ' + (passed ? '#2d5a2d' : C.boxStroke),
            borderRadius: 2,
          }}>
            {receiving ? 'Receiving' : '정상 통과'}
          </div>
        )}
      </div>
    );
  };

  /* 3. FocusLabel: "약효 확산 중" → "부작용 위험 감지" */
  window.FocusLabel = function(props) {
    var cfg = props.cfg, focusP = props.focusP;
    return (
      <div style={{
        position: 'absolute', top: 40, left: 0, right: 0,
        textAlign: 'center', opacity: focusP,
        pointerEvents: 'none', zIndex: 70,
        fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif',
      }}>
        <div style={{
          fontSize: 11, letterSpacing: '0.4em', color: C.boxStroke,
          textTransform: 'uppercase', fontWeight: 600, marginBottom: 8,
        }}>
          부작용 위험 · {cfg.sub}
        </div>
        <div style={{ fontSize: 30, fontWeight: 700, color: C.redDeep, letterSpacing: '0.04em' }}>
          {cfg.label}에서 부작용 위험 감지
        </div>
      </div>
    );
  };

  /* 4. SequenceHud: 위험 장기는 빨간색, 안전 장기는 초록색 */
  window.SequenceHud = function(props) {
    var t = props.t;
    var stages = [
      { key: 'intake',    label: '복용',  at: T.pillEnter.s,                    riskKey: null        },
      { key: 'intestine', label: '장',    at: T.intestActivate + SUB.spread.s,  riskKey: 'intestine' },
      { key: 'liver',     label: '간',    at: T.liverActivate  + SUB.spread.s,  riskKey: 'liver'     },
      { key: 'blood',     label: '혈액',  at: T.bloodActivate  + SUB.spread.s,  riskKey: 'blood'     },
      { key: 'joint',     label: '관절',  at: T.jointActivate  + SUB.spread.s,  riskKey: 'joint'     },
    ];

    var activeIdx = -1;
    for (var i = 0; i < stages.length; i++) {
      if (t >= stages[i].at) activeIdx = i;
    }

    return (
      <div style={{
        position: 'absolute', bottom: 28, left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex', alignItems: 'center', gap: 16,
        padding: '12px 22px',
        background: 'rgba(255,255,255,0.88)',
        backdropFilter: 'blur(8px)',
        borderRadius: 999,
        border: '1px solid ' + C.boxStroke + '22',
        fontFamily: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif',
        boxShadow: '0 6px 24px rgba(80,30,10,0.08)',
        zIndex: 80,
        whiteSpace: 'nowrap',
      }}>
        {stages.map(function(s, i) {
          var isActive = i === activeIdx;
          var isPast   = i < activeIdx;
          var atRisk   = s.riskKey ? !!__RISK[s.riskKey] : false;
          var safeColor = '#2E7D32', safeDeep = '#1B5E20';
          var color = isActive
            ? (atRisk ? C.red    : safeColor)
            : isPast
              ? (atRisk ? C.redDeep : safeDeep)
              : '#bdb6ac';

          return (
            <React.Fragment key={s.key}>
              {i > 0 && (
                <div style={{
                  width: 22, height: 1,
                  background: (isPast || isActive)
                    ? (atRisk ? C.redDeep : safeDeep)
                    : '#cfc8be',
                  transition: 'background 200ms',
                }} />
              )}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  width: 12, height: 12, borderRadius: 6,
                  background: (isActive || isPast) ? color : 'transparent',
                  border: '2px solid ' + color,
                  boxShadow: isActive ? '0 0 0 6px ' + color + '22' : 'none',
                  transition: 'box-shadow 200ms',
                }} />
                <div style={{
                  fontSize: 15,
                  fontWeight: isActive ? 700 : 500,
                  color: isActive ? color : isPast ? color : '#897f72',
                }}>
                  {s.label}{atRisk && (isPast || isActive) ? ' ⚠' : (isPast || isActive) ? ' ✓' : ''}
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    );
  };

})();
</script>"""


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
    dp    = DRUGS[drug]
    toxic = dp['toxic_cmax_mg_per_L']

    def _state(cmax):
        r = cmax / toxic
        if r >= 0.80: return 'danger'
        if r >= 0.50: return 'warn'
        return 'safe'

    return {
        '혈액': _state(sim['Cmax_blood']),
        '간':   _state(float(max(sim['C_liver']))),
        '활막': _state(sim['Cmax_tissue']),
    }


# ── 애니메이션 HTML 생성 ─────────────────────────────────────────────────────

def build_pathway_animation_html(sim, drug):
    """standalone HTML의 template에 환자 위험 데이터를 주입해 반환한다."""
    global _HTML_BASE
    if not _ANIM_PATH.exists():
        return "<p style='color:red;padding:20px;'>애니메이션 파일을 찾을 수 없습니다.</p>"

    if _HTML_BASE is None:
        _HTML_BASE = _ANIM_PATH.read_text(encoding='utf-8')

    dp    = DRUGS[drug]
    toxic = dp['toxic_cmax_mg_per_L']

    def _at_risk(cmax):
        return bool((cmax / toxic) >= 0.50)

    risk_data = {
        'intestine': False,                              # 장: 흡수 구획, 독성 표적 아님
        'liver':     _at_risk(float(max(sim['C_liver']))),
        'blood':     _at_risk(float(sim['Cmax_blood'])),
        'joint':     _at_risk(float(sim['Cmax_tissue'])),
    }

    # template JSON 추출
    m = re.search(
        r'(<script type="__bundler/template">)(.*?)(</script>)',
        _HTML_BASE, re.DOTALL
    )
    if not m:
        return _HTML_BASE  # fallback

    template_str = json.loads(m.group(2).strip())

    # 주입 코드 조립
    data_script = '<script>window.__patientRisk = ' + json.dumps(risk_data) + ';</script>\n'
    injection   = data_script + _PATHWAY_OVERRIDE_BABEL + '\n'

    # 삽입 위치: 마지막 인라인 App 스크립트 바로 앞
    marker = '<script type="text/babel">\nfunction App()'
    idx    = template_str.find(marker)
    if idx < 0:
        # 줄바꿈 차이 대비 fallback
        marker = '<script type="text/babel">function App()'
        idx    = template_str.find(marker)
    if idx < 0:
        # 최후 수단: 마지막 text/babel 스크립트 앞
        idx = template_str.rfind('<script type="text/babel">')

    if idx >= 0:
        template_str = template_str[:idx] + injection + template_str[idx:]

    # 재인코딩 후 HTML에 삽입
    new_json = json.dumps(template_str)
    new_html = _HTML_BASE[:m.start(2)] + new_json + _HTML_BASE[m.end(2):]
    return new_html


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
            min_value=int(dose_lo), max_value=int(dose_hi),
            value=int(dp['standard_dose_mg']), step=25,
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

    label, proba = predict_risk(
        models, drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype
    )
    organ_states = compute_risk_organs(sim, drug)

    # ────────────────────────────────────────────────────────────────────────
    # 영역 1 — 위험 등급 배너
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('① 나의 위험 등급')
    render_safety_banner(label)

    name_map  = {'dose_adjust': '주의\n(용량 조정)', 'standard': '안전\n(정상)', 'toxic': '위험\n(독성)'}
    color_map = {'standard': '#2E7D32', 'dose_adjust': '#E65100', 'toxic': '#B71C1C'}
    fig_prob  = go.Figure(go.Bar(
        x=[name_map.get(n, n) for n in label_names],
        y=proba,
        marker_color=[color_map.get(n, '#888') for n in label_names],
        text=[f'{p:.1%}' for p in proba],
        textposition='outside',
    ))
    fig_prob.update_layout(
        title='등급별 예측 확률',
        yaxis=dict(range=[0, 1.2], tickformat='.0%', title='확률'),
        xaxis_title='위험 등급', height=280,
        margin=dict(t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 2 — 혈중 농도-시간 곡선
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('② 시간에 따른 혈중 농도 변화')

    t = sim['t']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sim['C_blood'],  mode='lines', name='혈중 농도',
                             line=dict(color='crimson', width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=sim['C_tissue'], mode='lines', name='활막/말초 농도',
                             line=dict(color='steelblue', width=2.5)))
    fig.add_hline(y=dp['toxic_cmax_mg_per_L'],  line_dash='dash',     line_color='red',
                  annotation_text=f"독성 기준 {dp['toxic_cmax_mg_per_L']} mg/L",
                  annotation_position='top right')
    fig.add_hline(y=dp['adjust_cmax_mg_per_L'], line_dash='dot',      line_color='orange',
                  annotation_text=f"주의 기준 {dp['adjust_cmax_mg_per_L']} mg/L",
                  annotation_position='top right')
    fig.add_hline(y=dp['IC50_synovium_mg_per_L'], line_dash='longdash', line_color='steelblue',
                  annotation_text=f"절반 억제 농도 {dp['IC50_synovium_mg_per_L']} mg/L",
                  annotation_position='bottom right')
    fig.update_layout(
        title=f'{DRUG_KOREAN[drug]} 농도-시간 곡선 (24시간)',
        xaxis_title='시간 (h)', yaxis_title='농도 (mg/L)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        plot_bgcolor='rgba(250,250,250,1)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('최고 혈중 농도 (Cmax)',      f"{sim['Cmax_blood']:.3f} mg/L")
    c2.metric('최고 활막 농도',             f"{sim['Cmax_tissue']:.3f} mg/L")
    c3.metric('최고 농도 도달 시간 (Tmax)', f"{sim['Tmax_blood']:.2f} h")
    c4.metric('총 약물 노출량 (AUC₀₋₂₄)',   f"{sim['AUC_blood']:.1f} mg·h/L")

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 3 — 다른 복용량과 비교
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('③ 다른 복용량과 비교')

    compare_doses = sorted(set([int(dose_lo), int(dp['standard_dose_mg']), int(dose_hi), dose_mg]))
    cmp_rows = []
    for d in compare_doses:
        s = run_simulation(drug, float(d), float(body_weight), float(egfr), cyp2c9_genotype)
        if s['success']:
            lbl, _ = predict_risk(models, drug, float(d), float(body_weight), float(egfr), cyp2c9_genotype)
            badge_txt, _, _, _ = LABEL_KOREAN.get(lbl, LABEL_KOREAN['standard'])
            marker = ' ← 현재' if d == dose_mg else ''
            cmp_rows.append({
                '복용량 (mg)':    f'{d}{marker}',
                '최고 혈중 농도': f'{s["Cmax_blood"]:.3f} mg/L',
                '총 약물 노출량': f'{s["AUC_blood"]:.1f} mg·h/L',
                '위험 등급':      badge_txt,
            })

    if cmp_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 4 — 약물 경로 애니메이션 (환자 데이터 반영)
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('④ 약이 몸속을 이동하는 모습')

    # 장기 상태 뱃지
    state_color = {'safe': '#2E7D32', 'warn': '#E65100', 'danger': '#B71C1C'}
    state_label = {'safe': '안전 ✓', 'warn': '주의 ⚠', 'danger': '위험 ⚠'}
    organ_col_labels = [('혈액', organ_states['혈액']), ('간', organ_states['간']), ('활막/관절', organ_states['활막'])]
    badge_cols = st.columns(len(organ_col_labels))
    for col, (organ, state) in zip(badge_cols, organ_col_labels):
        col.markdown(
            f"<div style='text-align:center;background:{state_color[state]}20;"
            f"border:2px solid {state_color[state]};border-radius:8px;padding:10px;'>"
            f"<b style='color:{state_color[state]};font-size:1.1rem;'>{organ}</b><br>"
            f"<span style='color:{state_color[state]};font-size:0.95rem;'>{state_label[state]}</span></div>",
            unsafe_allow_html=True,
        )

    st.components.v1.html(
        build_pathway_animation_html(sim, drug),
        height=700,
    )

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 5 — 모델 설명
    # ────────────────────────────────────────────────────────────────────────
    st.subheader('⑤ 이 도구는 어떻게 작동하나요?')
    with st.expander('자세히 보기', expanded=False):
        st.markdown(f"""
### 작동 원리
1. **생리기반 약동학 모델 (PBPK)** — 약물이 장→간→혈액→활막으로 이동하는 과정을 수학 방정식으로 실시간 계산합니다.
2. **가상환자 500명 시뮬레이션** — 다양한 체중·신장 기능·효소 유형을 가진 가상 환자를 만들어 학습 데이터를 생성했습니다.
3. **머신러닝 분류기 (Random Forest)** — 5-겹 교차검증 F1-macro 0.79를 달성한 모델이 위험 등급을 예측합니다.

### 애니메이션 읽는 법
- **빨간 확산 효과** — 해당 장기에서 약물 농도가 독성 기준의 50% 이상으로 부작용 위험이 감지됩니다.
- **초록 테두리 + "정상 통과"** — 해당 장기는 안전 범위 안에 있습니다.
- **하단 HUD** — 각 단계에서 ✓(안전) 또는 ⚠(위험) 표시가 나타납니다.

### 용어 설명
| 용어 | 설명 |
|------|------|
| 최고 혈중 농도 (Cmax) | 복용 후 혈액 속 약물 농도의 최대값 |
| 총 약물 노출량 (AUC) | 24시간 동안 혈액이 약물에 노출된 총량 |
| 신장 기능 지표 (eGFR) | 신장이 1분에 혈액을 얼마나 걸러내는지의 속도 |
| 약물 분해 효소 유형 (CYP2C9) | 간에서 이 약물을 분해하는 효소의 종류 |
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

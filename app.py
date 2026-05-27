"""나의 약물 메디컬 트윈 — Streamlit 웹 앱 v2 (ko/en 다국어 지원)."""

import streamlit as st

# 언어 선택은 최초 렌더링 전에 결정되어야 하므로
# set_page_config 이전에 session_state에서 읽는다.
_DEFAULT_LANG = 'ko'
if 'language' not in st.session_state:
    st.session_state['language'] = _DEFAULT_LANG

st.set_page_config(
    page_title='나의 약물 메디컬 트윈 / My Drug Medical Twin',
    page_icon='💊',
    layout='wide',
)

import json
import re
import sys
from pathlib import Path

from translations import TRANSLATIONS

import numpy as np
import plotly.graph_objects as go
import joblib

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from params import DRUGS, CYP2C9_SCALING
from pbpk_model import simulate_pbpk
from egfr_calc import calculate_egfr_ckdepi

# PDF 리포트 생성 (fpdf2 필요; 없으면 PDF 버튼을 비활성화)
try:
    from report_generator import generate_pdf_report, _resolve_korean_font
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

_MODEL_DIR  = _ROOT / "models"
_ANIM_PATH  = _ROOT / "Medical Twin - Drug Pathway (Standalone).html"
_HTML_BASE  = None   # module-level HTML cache (read once per process)


_CYP2C9_ORDER = {'*1/*1': 0, '*1/*2': 1, '*1/*3': 2, '*2/*3': 3, '*3/*3': 4}

# ── Babel 오버라이드 스크립트 (모듈 상수) ────────────────────────────────────
# 번들러 template에 주입된다. 원본 JS globals(OrganPanel/getCurrentFocus 등)을
# 환자 위험 데이터 기반으로 덮어쓴다.

_PATHWAY_OVERRIDE_BABEL = r"""<script type="text/babel">
(function() {
  var __RISK   = window.__patientRisk   || {};
  var __LABELS = window.__patientLabels || {
    sideEffectRisk:   '부작용 위험',
    sideEffectPrefix: '',
    sideEffectSuffix: '에서 부작용 위험 감지',
    stages:    ['복용', '장', '간', '혈액', '관절'],
    safePass:  '정상 통과',
    receiving: '수신 중',
  };

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
            {receiving ? __LABELS.receiving : __LABELS.safePass}
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
          {__LABELS.sideEffectRisk} · {cfg.sub}
        </div>
        <div style={{ fontSize: 30, fontWeight: 700, color: C.redDeep, letterSpacing: '0.04em' }}>
          {__LABELS.sideEffectPrefix}{cfg.label}{__LABELS.sideEffectSuffix}
        </div>
      </div>
    );
  };

  /* 4. SequenceHud: 위험 장기는 빨간색, 안전 장기는 초록색 */
  window.SequenceHud = function(props) {
    var t = props.t;
    var stages = [
      { key: 'intake',    label: __LABELS.stages[0], at: T.pillEnter.s,                   riskKey: null        },
      { key: 'intestine', label: __LABELS.stages[1], at: T.intestActivate + SUB.spread.s, riskKey: 'intestine' },
      { key: 'liver',     label: __LABELS.stages[2], at: T.liverActivate  + SUB.spread.s, riskKey: 'liver'     },
      { key: 'blood',     label: __LABELS.stages[3], at: T.bloodActivate  + SUB.spread.s, riskKey: 'blood'     },
      { key: 'joint',     label: __LABELS.stages[4], at: T.jointActivate  + SUB.spread.s, riskKey: 'joint'     },
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
def run_simulation(drug, dose_mg, body_weight, egfr, cyp2c9_genotype, age=40):
    return simulate_pbpk(drug, dose_mg, body_weight, egfr, cyp2c9_genotype, age=age)


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
        'blood':    _state(sim['Cmax_blood']),
        'liver':    _state(float(max(sim['C_liver']))),
        'synovium': _state(sim['Cmax_tissue']),
    }


# ── 애니메이션 HTML 생성 ─────────────────────────────────────────────────────

def build_pathway_animation_html(sim, drug, lang: str = 'ko'):
    """standalone HTML의 template에 환자 위험 데이터와 다국어 레이블을 주입해 반환한다."""
    global _HTML_BASE
    TR = TRANSLATIONS[lang]

    if not _ANIM_PATH.exists():
        return TR['anim_file_not_found']

    import os
    if _HTML_BASE is None or os.environ.get('MT_DISABLE_HTML_CACHE') == '1':
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

    # 다국어 레이블 조립
    anim_labels = {
        'sideEffectRisk':   TR['anim_side_effect_risk'],
        'sideEffectPrefix': TR['anim_side_effect_prefix'],
        'sideEffectSuffix': TR['anim_side_effect_suffix'],
        'stages':           TR['anim_stages'],
        'safePass':         TR['anim_safe_pass'],
        'receiving':        TR['anim_receiving'],
    }

    # 주입 코드 조립
    data_script = (
        '<script>\n'
        'window.__patientRisk = '   + json.dumps(risk_data)   + ';\n'
        'window.__patientLabels = ' + json.dumps(anim_labels) + ';\n'
        '</script>\n'
    )
    injection = data_script + _PATHWAY_OVERRIDE_BABEL + '\n'

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
    # HTML 파서가 script 태그를 조기 종료하지 못하도록 </ 시퀀스 이스케이프
    # JSON 명세상 "<\/script>" 와 "</script>" 는 동등하나, 브라우저 HTML 파서는
    # raw text에서 </script>를 만나면 script 블록을 닫아 버린다.
    new_json = new_json.replace('</', '<\\/')
    new_html = _HTML_BASE[:m.start(2)] + new_json + _HTML_BASE[m.end(2):]
    return new_html


# ── 안전 배너 ────────────────────────────────────────────────────────────────

def render_safety_banner(label, TR: dict):
    rl = TR['risk_labels']
    badge, color, bg, msg = rl.get(label, rl['standard'])
    st.markdown(
        f"""
        <div style="background:{bg};border-left:6px solid {color};
             border-radius:8px;padding:18px 24px;margin-bottom:14px;">
          <h2 style="color:{color};margin:0 0 6px 0;">{TR['risk_banner_prefix']}{badge}</h2>
          <p style="font-size:1.05rem;margin:0;color:#333;">{msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 예측 ─────────────────────────────────────────────────────────────────────

def predict_risk(models, drug, dose_mg, body_weight, egfr, cyp2c9_genotype, age=40):
    drug_enc_val   = int(models['drug_enc'].transform([drug])[0])
    cyp2c9_enc_val = _CYP2C9_ORDER[cyp2c9_genotype]
    # ★ 순서: ml_model.py _FEATURE_COLS와 반드시 일치
    # [body_weight, age, egfr, cyp2c9_enc, drug_enc, dose_mg]
    X_raw = np.array([[body_weight, age, egfr, cyp2c9_enc_val, drug_enc_val, dose_mg]])
    X_sc  = models['scaler'].transform(X_raw)
    pred_idx = models['rf'].predict(X_sc)[0]
    proba    = models['rf'].predict_proba(X_sc)[0]
    label    = models['label_enc'].inverse_transform([pred_idx])[0]
    return label, proba


# ── 수치 설명 섹션 ───────────────────────────────────────────────────────────

def render_guide_section():
    """비전문가를 위한 eGFR·CYP2C9 설명 섹션."""
    import pandas as pd

    # ─── 1. eGFR 카드 ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="border:2px solid #42a5f5;border-radius:12px;padding:20px 24px;margin-bottom:4px;">
<h3 style="color:#1565c0;margin-top:0;">🫘 eGFR — 신장이 얼마나 잘 걸러주는가</h3>
<p><strong>신장은 몸속의 정수기입니다. eGFR은 이 정수기 필터가 얼마나 깨끗하게 작동하는지를 숫자로 나타냅니다.</strong></p>
<p>약을 먹으면 혈액 속으로 흡수되어 온몸을 돌다가, 결국 신장을 통해 소변으로 빠져나갑니다.<br>
신장 기능이 떨어지면 약이 몸속에 더 오래 머물러 쌓이고, 이는 독성으로 이어질 수 있습니다.<br>
eGFR 수치가 낮을수록 약의 용량을 줄이거나 주의가 필요합니다.</p>
<svg width="100%" viewBox="0 0 400 130" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;display:block;margin:12px auto;">
  <!-- 건강한 신장 (왼쪽) -->
  <g transform="translate(30,5) scale(0.55)">
    <path d="M 50,5 C 75,5 95,22 93,52 C 91,82 74,108 52,108 C 35,108 16,93 12,72 C 8,51 16,37 28,30 C 16,23 18,8 33,5 C 39,3 44,5 50,5 Z" fill="#ef9a9a"/>
    <ellipse cx="24" cy="54" rx="9" ry="16" fill="#fce4ec"/>
  </g>
  <text x="62" y="84" text-anchor="middle" font-size="12" font-weight="bold" fill="#c62828" font-family="sans-serif">건강한 신장</text>
  <text x="62" y="100" text-anchor="middle" font-size="11" fill="#666" font-family="sans-serif">신기능 지표(eGFR) ≥ 60</text>
  <!-- 화살표 -->
  <text x="200" y="45" text-anchor="middle" font-size="30" fill="#aaa" font-family="sans-serif">→</text>
  <text x="200" y="63" text-anchor="middle" font-size="11" fill="#888" font-family="sans-serif">기능 저하</text>
  <!-- 기능 저하된 신장 (오른쪽) -->
  <g transform="translate(245,5) scale(0.55)">
    <path d="M 50,5 C 75,5 95,22 93,52 C 91,82 74,108 52,108 C 35,108 16,93 12,72 C 8,51 16,37 28,30 C 16,23 18,8 33,5 C 39,3 44,5 50,5 Z" fill="#b0bec5"/>
    <ellipse cx="24" cy="54" rx="9" ry="16" fill="#eceff1"/>
    <line x1="55" y1="25" x2="68" y2="38" stroke="#78909c" stroke-width="2"/>
    <line x1="60" y1="60" x2="72" y2="72" stroke="#78909c" stroke-width="2"/>
  </g>
  <text x="277" y="84" text-anchor="middle" font-size="12" font-weight="bold" fill="#546e7a" font-family="sans-serif">기능 저하된 신장</text>
  <text x="277" y="100" text-anchor="middle" font-size="11" fill="#666" font-family="sans-serif">신기능 지표(eGFR) &lt; 30</text>
</svg>
</div>
""", unsafe_allow_html=True)

    _egfr_colors = ['#E8F5E9', '#FFF9C4', '#FFE0B2', '#FFEBEE']
    egfr_df = pd.DataFrame([
        {'단계': '정상',        '신기능 지표(eGFR)': '60 이상', '의미': '신장이 잘 기능함',    '권고사항': '일반 용량 적용 가능'},
        {'단계': '경도 저하',   '신기능 지표(eGFR)': '45 ~ 59', '의미': '약간 주의 필요',     '권고사항': '경미한 모니터링'},
        {'단계': '중등도 저하', '신기능 지표(eGFR)': '30 ~ 44', '의미': '용량 조정 고려',     '권고사항': '의사와 상담'},
        {'단계': '중증 저하',   '신기능 지표(eGFR)': '30 미만',  '의미': '전문의 상담 필수', '권고사항': '용량 감량 또는 금기'},
    ])

    def _color_egfr_row(row):
        return [f'background-color:{_egfr_colors[row.name]};color:#222'] * len(row)

    st.dataframe(
        egfr_df.style.apply(_color_egfr_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # ─── 2. CYP2C9 카드 ───────────────────────────────────────────────────────
    st.markdown("""
<div style="border:2px solid #ab47bc;border-radius:12px;padding:20px 24px;margin-bottom:4px;">
<h3 style="color:#6a1b9a;margin-top:0;">⚗️ CYP2C9 — 간이 약을 분해하는 속도</h3>
<p><strong>간은 약을 분해하는 공장입니다. CYP2C9는 그 공장 안의 분해 효소, 즉 가위입니다.<br>
사람마다 이 가위의 성능이 다르게 태어납니다.</strong></p>
<p>CYP2C9는 이부프로펜, 나프록센 등 소염진통제를 분해하는 간 효소입니다.<br>
유전자 타입에 따라 약을 분해하는 속도가 크게 달라집니다.<br>
분해가 느린 유전형(*3/*3)을 가진 분은 같은 약을 먹어도 혈중 농도가 훨씬 높게 오를 수 있습니다.</p>
<svg width="100%" viewBox="0 0 400 160" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;display:block;margin:12px auto;">
  <text x="200" y="18" text-anchor="middle" font-size="12" fill="#777" font-family="sans-serif">가위가 클수록 약을 빠르게 분해합니다</text>
  <!-- 큰 가위 (*1/*1) -->
  <g transform="translate(75,80)">
    <ellipse cx="-22" cy="-28" rx="16" ry="13" fill="none" stroke="#1565c0" stroke-width="3"/>
    <ellipse cx="-22" cy="28" rx="16" ry="13" fill="none" stroke="#1565c0" stroke-width="3"/>
    <line x1="-7" y1="-22" x2="42" y2="10" stroke="#1565c0" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="-7" y1="22" x2="42" y2="-10" stroke="#1565c0" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="16" cy="0" r="5" fill="#1565c0"/>
  </g>
  <text x="75" y="125" text-anchor="middle" font-size="13" font-weight="bold" fill="#1565c0" font-family="sans-serif">*1/*1</text>
  <text x="75" y="142" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">빠름 (기준)</text>
  <!-- 중간 가위 (*1/*3) -->
  <g transform="translate(200,80) scale(0.76)">
    <ellipse cx="-22" cy="-28" rx="16" ry="13" fill="none" stroke="#7b1fa2" stroke-width="3"/>
    <ellipse cx="-22" cy="28" rx="16" ry="13" fill="none" stroke="#7b1fa2" stroke-width="3"/>
    <line x1="-7" y1="-22" x2="42" y2="10" stroke="#7b1fa2" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="-7" y1="22" x2="42" y2="-10" stroke="#7b1fa2" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="16" cy="0" r="5" fill="#7b1fa2"/>
  </g>
  <text x="200" y="125" text-anchor="middle" font-size="13" font-weight="bold" fill="#7b1fa2" font-family="sans-serif">*1/*3</text>
  <text x="200" y="142" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">느림</text>
  <!-- 작은 가위 (*3/*3) -->
  <g transform="translate(325,80) scale(0.52)">
    <ellipse cx="-22" cy="-28" rx="16" ry="13" fill="none" stroke="#90a4ae" stroke-width="3"/>
    <ellipse cx="-22" cy="28" rx="16" ry="13" fill="none" stroke="#90a4ae" stroke-width="3"/>
    <line x1="-7" y1="-22" x2="42" y2="10" stroke="#90a4ae" stroke-width="3.5" stroke-linecap="round"/>
    <line x1="-7" y1="22" x2="42" y2="-10" stroke="#90a4ae" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="16" cy="0" r="5" fill="#90a4ae"/>
  </g>
  <text x="325" y="125" text-anchor="middle" font-size="13" font-weight="bold" fill="#90a4ae" font-family="sans-serif">*3/*3</text>
  <text x="325" y="142" text-anchor="middle" font-size="11" fill="#555" font-family="sans-serif">극히 느림</text>
</svg>
</div>
""", unsafe_allow_html=True)

    _cyp_colors = ['#E3F2FD', '#EDE7F6', '#EDE7F6', '#FCE4EC', '#FFEBEE']
    cyp_df = pd.DataFrame([
        {'유전형': '*1/*1', '별명': '정상 대사자', '분해 속도': '빠름 (기준)', '권고사항': '일반 용량 적용 가능'},
        {'유전형': '*1/*2', '별명': '중간 대사자', '분해 속도': '약간 느림',   '권고사항': '경미한 주의'},
        {'유전형': '*1/*3', '별명': '중간 대사자', '분해 속도': '느림',        '권고사항': '용량 주의'},
        {'유전형': '*2/*3', '별명': '느린 대사자', '분해 속도': '매우 느림',   '권고사항': '용량 감량 고려'},
        {'유전형': '*3/*3', '별명': '불량 대사자', '분해 속도': '극히 느림',   '권고사항': '전문의 상담 필수'},
    ])

    def _color_cyp_row(row):
        return [f'background-color:{_cyp_colors[row.name]};color:#222'] * len(row)

    st.dataframe(
        cyp_df.style.apply(_color_cyp_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # ─── 3. 왜 이 두 가지가 중요한가 ──────────────────────────────────────────
    st.info(
        "**이 시뮬레이터는 여러분의 신장 기능(eGFR)과 유전자 타입(CYP2C9)을 입력받아, "
        "약이 몸 안에서 어떻게 분포하는지를 계산합니다.**  \n\n"
        "특히 신기능이 낮거나, 약을 분해하는 속도가 느린 분은  \n"
        "표준 처방 가이드라인만으로는 충분한 보호를 받지 못할 수 있습니다.  \n"
        "이 도구는 그런 분들을 위해 만들어졌습니다."
    )


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    # ── 언어 선택 (사이드바 최상단) ──────────────────────────────────────────
    with st.sidebar:
        lang_raw = st.radio(
            '',
            options=['🇰🇷 한국어', '🇺🇸 English'],
            index=0 if st.session_state.get('language', 'ko') == 'ko' else 1,
            horizontal=True,
            label_visibility='collapsed',
            key='language_radio',
        )
        lang = 'ko' if '한국어' in lang_raw else 'en'
        st.session_state['language'] = lang
    TR = TRANSLATIONS[lang]

    # ── 페이지 헤더 ───────────────────────────────────────────────────────────
    st.title(TR['app_title'])
    st.caption(TR['app_caption'])

    models = load_models()
    if models is None:
        st.error(TR['model_not_found_error'])
        st.stop()

    label_names = list(models['label_enc'].classes_)

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header(TR['sidebar_header'])

        drug = st.selectbox(
            TR['drug_select_label'],
            options=list(DRUGS.keys()),
            format_func=lambda x: f"{TR['drug_names'][x]} ({x.capitalize()})",
        )
        dp = DRUGS[drug]
        dose_lo, dose_hi = dp['dose_range']
        dose_mg = st.slider(
            TR['dose_label'],
            min_value=int(dose_lo), max_value=int(dose_hi),
            value=int(dp['standard_dose_mg']), step=25,
        )
        body_weight = st.slider(TR['weight_label'], 35, 130, 65, 1)
        age = st.slider(TR.get('age_label', '나이 (세)'), 1, 95, 40, 1)
        egfr_mode = st.radio(
            TR.get('egfr_mode_label', 'eGFR 입력 방식'),
            options=['direct', 'creatinine'],
            format_func=lambda x: (
                TR.get('egfr_mode_direct', 'eGFR 직접 입력')
                if x == 'direct'
                else TR.get('egfr_mode_calc', '크레아티닌으로 자동 계산')
            ),
            horizontal=True,
        )
        if egfr_mode == 'direct':
            egfr = float(st.slider(
                TR['egfr_label'],
                min_value=15, max_value=140, value=110, step=1,
                help=TR['egfr_help'],
            ))
        else:
            cr_val = st.number_input(
                TR.get('cr_label', '혈청 크레아티닌 (mg/dL)'),
                min_value=0.3, max_value=15.0, value=1.0, step=0.1,
                format='%.1f',
            )
            cr_sex = st.radio(
                TR.get('cr_sex_label', '성별'),
                options=['female', 'male'],
                format_func=lambda x: (
                    TR.get('cr_sex_female', '여성') if x == 'female'
                    else TR.get('cr_sex_male', '남성')
                ),
                horizontal=True,
            )
            # 나이는 위에서 입력한 age 슬라이더 값을 재사용
            egfr = calculate_egfr_ckdepi(float(cr_val), int(age), cr_sex)
            # 신기능 단계 레이블 결정
            if egfr >= 60:
                _stage = TR.get('egfr_stage_normal', '정상')
            elif egfr >= 45:
                _stage = TR.get('egfr_stage_mild', '경도 저하')
            elif egfr >= 30:
                _stage = TR.get('egfr_stage_moderate', '중등도 저하')
            else:
                _stage = TR.get('egfr_stage_severe', '중증 저하')
            st.success(
                f"**{TR.get('egfr_calc_result', '계산된 eGFR')}:** {egfr} mL/min/1.73m²  \n"
                f"({_stage})"
            )
        cyp2c9_genotype = st.selectbox(
            TR['cyp_label'],
            options=list(CYP2C9_SCALING.keys()),
            index=0,
            format_func=lambda x: TR['cyp2c9_labels'][x],
        )
        st.divider()
        st.caption(
            f"**{TR['drug_names'][drug]}**  \n"
            f"{TR['caution_threshold']}: {dp['adjust_cmax_mg_per_L']} mg/L  \n"
            f"{TR['toxic_threshold']}: {dp['toxic_cmax_mg_per_L']} mg/L"
        )
        if not dp['cyp2c9_dependent']:
            st.info(TR['apap_no_cyp_info'])

    # ── 시뮬레이션 & 예측 ────────────────────────────────────────────────────
    sim = run_simulation(drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype, int(age))
    if not sim['success']:
        st.error(TR['sim_error'])
        st.stop()

    label, proba = predict_risk(
        models, drug, float(dose_mg), float(body_weight), float(egfr), cyp2c9_genotype, int(age)
    )
    organ_states = compute_risk_organs(sim, drug)

    # ────────────────────────────────────────────────────────────────────────
    # 영역 ① 위험 등급 배너
    # ────────────────────────────────────────────────────────────────────────
    st.subheader(TR['section1_header'])
    render_safety_banner(label, TR)

    # 65세 이상 + 용량조정·독성 위험 시 고령자 특화 경고
    if age >= 65 and label in ('dose_adjust', 'toxic'):
        st.warning(TR.get(
            'elderly_warning',
            '65세 이상 고령자는 약물 배설이 느려 부작용 위험이 더 높습니다. 특별히 주의하세요.',
        ))

    color_map = {'standard': '#2E7D32', 'dose_adjust': '#E65100', 'toxic': '#B71C1C'}
    fig_prob = go.Figure(go.Bar(
        x=[TR['risk_bar_labels'].get(n, n) for n in label_names],
        y=proba,
        marker_color=[color_map.get(n, '#888') for n in label_names],
        text=[f'{p:.1%}' for p in proba],
        textposition='outside',
    ))
    fig_prob.update_layout(
        title=TR['prob_chart_title'],
        yaxis=dict(range=[0, 1.2], tickformat='.0%', title=TR['prob_chart_y']),
        xaxis_title=TR['prob_chart_x'], height=280,
        margin=dict(t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 ② 혈중 농도-시간 곡선
    # ────────────────────────────────────────────────────────────────────────
    st.subheader(TR['section2_header'])
    t = sim['t']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sim['C_blood'],  mode='lines', name=TR['trace_blood'],
                             line=dict(color='crimson', width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=sim['C_tissue'], mode='lines', name=TR['trace_tissue'],
                             line=dict(color='steelblue', width=2.5)))
    fig.add_hline(y=dp['toxic_cmax_mg_per_L'], line_dash='dash', line_color='red',
                  annotation_text=f"{TR['annot_toxic']} {dp['toxic_cmax_mg_per_L']} mg/L",
                  annotation_position='top right')
    fig.add_hline(y=dp['adjust_cmax_mg_per_L'], line_dash='dot', line_color='orange',
                  annotation_text=f"{TR['annot_caution']} {dp['adjust_cmax_mg_per_L']} mg/L",
                  annotation_position='top right')
    fig.add_hline(y=dp['IC50_synovium_mg_per_L'], line_dash='longdash', line_color='steelblue',
                  annotation_text=f"{TR['annot_ic50']} {dp['IC50_synovium_mg_per_L']} mg/L",
                  annotation_position='bottom right')
    fig.update_layout(
        title=f"{TR['drug_names'][drug]} {TR['chart_conc_title']}",
        xaxis_title=TR['chart_x_time'], yaxis_title=TR['chart_y_conc'],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        plot_bgcolor='rgba(250,250,250,1)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(TR['metric_cmax_blood'],  f"{sim['Cmax_blood']:.3f} mg/L")
    c2.metric(TR['metric_cmax_tissue'], f"{sim['Cmax_tissue']:.3f} mg/L")
    c3.metric(TR['metric_tmax'],        f"{sim['Tmax_blood']:.2f} h")
    c4.metric(TR['metric_auc'],         f"{sim['AUC_blood']:.1f} mg·h/L")
    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 ② 하단: PDF 리포트 다운로드
    # ────────────────────────────────────────────────────────────────────────
    if _PDF_AVAILABLE:
        # 폰트 미설치 경고
        if _resolve_korean_font() is None:
            st.warning(
                '한글 폰트가 없어 PDF 생성이 제한됩니다. '
                'fonts/NanumGothic.ttf를 추가하세요.'
            )

        if st.button('리포트 생성하기', key='gen_report'):
            # ① plotly 차트 → PNG bytes (kaleido 필요)
            _chart_bytes = None
            try:
                _chart_bytes = fig.to_image(format='png', width=800, height=400, scale=2)
            except Exception:
                st.info('PDF용 차트 변환에 실패했습니다. kaleido 설치를 확인하세요.')

            # ② 환자 정보 딕셔너리
            _patient_info = {
                '약물':         f"{TR['drug_names'][drug]} ({drug.capitalize()})",
                '복용량':       f'{dose_mg} mg',
                '체중':         f'{body_weight} kg',
                '나이':         f'{age} 세',
                'eGFR':         f'{egfr:.0f} mL/min/1.73m²',
                'CYP2C9 유전형': cyp2c9_genotype,
            }

            # ③ PDF 생성
            try:
                _pdf_bytes = generate_pdf_report(
                    patient_info=_patient_info,
                    sim=sim,
                    label=label,
                    drug_korean=TR['drug_names'][drug],
                    chart_png_bytes=_chart_bytes,
                )
                st.session_state['pdf_bytes'] = _pdf_bytes
                st.session_state['pdf_drug']  = drug
            except Exception as _e:
                st.error(f'PDF 생성 실패: {_e}')

        # 다운로드 버튼 (세션에 PDF가 있으면 항상 표시)
        if 'pdf_bytes' in st.session_state:
            st.download_button(
                label='📄 나의 약물 리포트 PDF 다운로드',
                data=st.session_state['pdf_bytes'],
                file_name=f"medical_twin_report_{st.session_state['pdf_drug']}.pdf",
                mime='application/pdf',
            )

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 ③ 다른 복용량과 비교
    # ────────────────────────────────────────────────────────────────────────
    st.subheader(TR['section3_header'])
    compare_doses = sorted(set([int(dose_lo), int(dp['standard_dose_mg']), int(dose_hi), dose_mg]))
    cmp_rows = []
    for d in compare_doses:
        s = run_simulation(drug, float(d), float(body_weight), float(egfr), cyp2c9_genotype)
        if s['success']:
            lbl, _ = predict_risk(models, drug, float(d), float(body_weight), float(egfr), cyp2c9_genotype)
            rl = TR['risk_labels']
            badge_txt, _, _, _ = rl.get(lbl, rl['standard'])
            marker = TR['table_current_marker'] if d == dose_mg else ''
            cmp_rows.append({
                TR['table_col_dose']: f'{d}{marker}',
                TR['table_col_cmax']: f'{s["Cmax_blood"]:.3f} mg/L',
                TR['table_col_auc']:  f'{s["AUC_blood"]:.1f} mg·h/L',
                TR['table_col_risk']: badge_txt,
            })
    if cmp_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 ④ 약물 경로 애니메이션
    # ────────────────────────────────────────────────────────────────────────
    st.subheader(TR['section4_header'])
    state_color = {'safe': '#2E7D32', 'warn': '#E65100', 'danger': '#B71C1C'}
    state_label = {
        'safe':   TR['state_safe'],
        'warn':   TR['state_warn'],
        'danger': TR['state_danger'],
    }
    organ_display = [
        (TR['organ_blood'],    organ_states['blood']),
        (TR['organ_liver'],    organ_states['liver']),
        (TR['organ_synovium'], organ_states['synovium']),
    ]
    badge_cols = st.columns(len(organ_display))
    for col, (organ_name, state) in zip(badge_cols, organ_display):
        col.markdown(
            f"<div style='text-align:center;background:{state_color[state]}20;"
            f"border:2px solid {state_color[state]};border-radius:8px;padding:10px;'>"
            f"<b style='color:{state_color[state]};font-size:1.1rem;'>{organ_name}</b><br>"
            f"<span style='color:{state_color[state]};font-size:0.95rem;'>{state_label[state]}</span></div>",
            unsafe_allow_html=True,
        )
    st.components.v1.html(
        build_pathway_animation_html(sim, drug, lang),
        height=700,
    )
    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # 영역 ⑤ 모델 설명
    # ────────────────────────────────────────────────────────────────────────
    st.subheader(TR['section5_header'])
    with st.expander(TR['expander_label'], expanded=False):
        st.markdown(TR['expander_content'])

    # ────────────────────────────────────────────────────────────────────────
    # 📖 이 수치가 뭔가요?
    # ────────────────────────────────────────────────────────────────────────
    with st.expander('📖 이 수치가 뭔가요? — eGFR과 CYP2C9 쉽게 이해하기', expanded=False):
        render_guide_section()

    # ── 면책 문구 ─────────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown(
        f'<p style="text-align:center;color:#888;font-size:0.85rem;">{TR["disclaimer"]}</p>',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()

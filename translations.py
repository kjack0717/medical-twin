"""translations.py — Medical Twin 앱 다국어 번역 딕셔너리 (ko / en)."""

TRANSLATIONS: dict[str, dict] = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  한국어
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'ko': {
        # 페이지 공통
        'page_title':  '나의 약물 메디컬 트윈',
        'app_title':   '💊 나의 약물 메디컬 트윈',
        'app_caption': '생리기반 약동학 모델(PBPK) + 머신러닝으로 나에게 맞는 약물 위험도를 예측합니다',

        # 사이드바
        'sidebar_header': '내 정보 입력',
        'drug_select_label': '복용할 약물',
        'dose_label':   '복용량 (mg)',
        'weight_label': '체중 (kg)',
        'egfr_label':   '신장 기능 지표 (eGFR, mL/min/1.73m²)',
        'egfr_help':    '정상 90 이상 | 경도 저하 60-89 | 중등도 저하 30-59 | 중증 저하 15-29',
        'cyp_label':    '약물 분해 효소 유형 (CYP2C9)',
        'caution_threshold': '주의 기준 농도',
        'toxic_threshold':   '위험 기준 농도',
        'apap_no_cyp_info':  '아세트아미노펜은 약물 분해 효소 유형(CYP2C9)의 영향을 받지 않습니다.',

        # 약물 이름
        'drug_names': {
            'ibuprofen':     '이부프로펜',
            'naproxen':      '나프록센',
            'celecoxib':     '셀레콕시브',
            'acetaminophen': '아세트아미노펜',
        },

        # CYP2C9 유전형
        'cyp2c9_labels': {
            '*1/*1': '*1/*1 — 정상 분해 (약물이 빠르게 제거됨)',
            '*1/*2': '*1/*2 — 약간 느린 분해',
            '*1/*3': '*1/*3 — 느린 분해',
            '*2/*3': '*2/*3 — 매우 느린 분해 (약물 축적 주의)',
            '*3/*3': '*3/*3 — 거의 분해 안 됨 (가장 높은 축적 위험)',
        },

        # 위험 등급 (뱃지 텍스트, 테두리 색, 배경색, 설명 메시지)
        'risk_labels': {
            'standard':    ('안전 — 정상 범위',       '#2E7D32', '#E8F5E9',
                            '현재 복용량은 안전 범위 안에 있습니다.'),
            'dose_adjust': ('주의 — 용량 줄이기 권장', '#E65100', '#FFF3E0',
                            '혈중 농도가 높아질 수 있습니다. 복용량을 줄이거나 복용 간격을 늘리세요.'),
            'toxic':       ('위험 — 독성 가능성',      '#B71C1C', '#FFEBEE',
                            '이 조합은 독성 수준에 도달할 위험이 있습니다. 즉시 의사와 상담하세요.'),
        },

        # 상태/에러 메시지
        'model_not_found_error': '모델 파일을 찾을 수 없습니다. `python ml_model.py`를 먼저 실행해 주세요.',
        'sim_error':             '계산 오류가 발생했습니다. 체중·신장 기능·복용량 값을 바꿔 다시 시도해 주세요.',
        'anim_file_not_found':   "애니메이션 파일을 찾을 수 없습니다.",

        # 섹션 ① 위험 등급
        'section1_header':   '① 나의 위험 등급',
        'risk_banner_prefix': '예측 결과: ',
        'prob_chart_title':  '등급별 예측 확률',
        'prob_chart_y':      '확률',
        'prob_chart_x':      '위험 등급',
        'risk_bar_labels': {
            'standard':    '안전\n(정상)',
            'dose_adjust': '주의\n(용량 조정)',
            'toxic':       '위험\n(독성)',
        },

        # 섹션 ② 농도-시간 곡선
        'section2_header':   '② 시간에 따른 혈중 농도 변화',
        'trace_blood':       '혈중 농도',
        'trace_tissue':      '활막/말초 농도',
        'annot_toxic':       '독성 기준',
        'annot_caution':     '주의 기준',
        'annot_ic50':        '절반 억제 농도',
        'chart_conc_title':  '농도-시간 곡선 (24시간)',
        'chart_x_time':      '시간 (h)',
        'chart_y_conc':      '농도 (mg/L)',
        'metric_cmax_blood': '최고 혈중 농도 (Cmax)',
        'metric_cmax_tissue':'최고 활막 농도',
        'metric_tmax':       '최고 농도 도달 시간 (Tmax)',
        'metric_auc':        '총 약물 노출량 (AUC₀₋₂₄)',

        # 섹션 ③ 복용량 비교
        'section3_header':      '③ 다른 복용량과 비교',
        'table_col_dose':       '복용량 (mg)',
        'table_col_cmax':       '최고 혈중 농도',
        'table_col_auc':        '총 약물 노출량',
        'table_col_risk':       '위험 등급',
        'table_current_marker': ' ← 현재',

        # 섹션 ④ 애니메이션
        'section4_header': '④ 약이 몸속을 이동하는 모습',
        'organ_blood':    '혈액',
        'organ_liver':    '간',
        'organ_synovium': '활막/관절',
        'state_safe':   '안전 ✓',
        'state_warn':   '주의 ⚠',
        'state_danger': '위험 ⚠',

        # 섹션 ⑤ 모델 설명
        'section5_header':  '⑤ 이 도구는 어떻게 작동하나요?',
        'expander_label':   '자세히 보기',
        'expander_content': """\
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
""",

        # 면책 문구
        'disclaimer': (
            '⚠️ 본 도구는 고등학교 연구 목적의 교육용 시뮬레이터입니다. '
            '실제 임상 결정에 사용해서는 안 됩니다. '
            '의약품 복용 전 반드시 의사 또는 약사와 상담하세요.'
        ),

        # 애니메이션 오버라이드 레이블 (window.__patientLabels 로 전달)
        # {prefix}{장기명}{suffix} 패턴으로 FocusLabel 텍스트 구성
        'anim_side_effect_risk':   '부작용 위험',
        'anim_side_effect_prefix': '',
        'anim_side_effect_suffix': '에서 부작용 위험 감지',
        'anim_stages':   ['복용', '장', '간', '혈액', '관절'],
        'anim_safe_pass': '정상 통과',
        'anim_receiving': '수신 중',
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  English
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'en': {
        # Page common
        'page_title':  'My Drug Medical Twin',
        'app_title':   '💊 My Drug Medical Twin',
        'app_caption': 'Personalized drug toxicity risk prediction · PBPK simulation + Machine Learning',

        # Sidebar
        'sidebar_header': 'Enter Your Profile',
        'drug_select_label': 'Drug',
        'dose_label':   'Dose (mg)',
        'weight_label': 'Body Weight (kg)',
        'egfr_label':   'Kidney Function (eGFR, mL/min/1.73m²)',
        'egfr_help':    'Normal ≥ 90 | Mild reduction 60–89 | Moderate 30–59 | Severe 15–29',
        'cyp_label':    'Metabolizing Enzyme Type (CYP2C9)',
        'caution_threshold': 'Caution threshold',
        'toxic_threshold':   'Toxic threshold',
        'apap_no_cyp_info':  (
            'Acetaminophen is metabolized by CYP2E1, not CYP2C9. '
            'The enzyme type setting has no effect on its simulation.'
        ),

        # Drug names
        'drug_names': {
            'ibuprofen':     'Ibuprofen',
            'naproxen':      'Naproxen',
            'celecoxib':     'Celecoxib',
            'acetaminophen': 'Acetaminophen',
        },

        # CYP2C9 genotypes
        'cyp2c9_labels': {
            '*1/*1': '*1/*1 — Normal metabolism (drug cleared quickly)',
            '*1/*2': '*1/*2 — Slightly slow metabolism',
            '*1/*3': '*1/*3 — Slow metabolism',
            '*2/*3': '*2/*3 — Very slow metabolism (watch for drug accumulation)',
            '*3/*3': '*3/*3 — Minimal metabolism (highest accumulation risk)',
        },

        # Risk labels (badge text, border color, bg color, description)
        'risk_labels': {
            'standard':    ('Safe — Within Normal Range',       '#2E7D32', '#E8F5E9',
                            'Your current dose is within the safe range.'),
            'dose_adjust': ('Caution — Dose Reduction Advised', '#E65100', '#FFF3E0',
                            'Blood concentration may rise. Consider reducing dose or increasing dosing interval.'),
            'toxic':       ('Danger — Toxicity Risk',           '#B71C1C', '#FFEBEE',
                            'This combination carries a risk of reaching toxic levels. Consult a doctor immediately.'),
        },

        # Status/error messages
        'model_not_found_error': 'Model files not found. Please run `python ml_model.py` first.',
        'sim_error':             'Simulation error. Try adjusting body weight, kidney function, or dose.',
        'anim_file_not_found':   "Animation file not found.",

        # Section ① Risk level
        'section1_header':    '① Risk Level',
        'risk_banner_prefix': 'Prediction: ',
        'prob_chart_title':   'Predicted Probability by Risk Category',
        'prob_chart_y':       'Probability',
        'prob_chart_x':       'Risk Category',
        'risk_bar_labels': {
            'standard':    'Safe\n(Normal)',
            'dose_adjust': 'Caution\n(Adjust)',
            'toxic':       'Danger\n(Toxic)',
        },

        # Section ② Concentration-time curve
        'section2_header':    '② Blood Concentration Over Time',
        'trace_blood':        'Blood Concentration',
        'trace_tissue':       'Synovium / Peripheral Concentration',
        'annot_toxic':        'Toxic threshold',
        'annot_caution':      'Caution threshold',
        'annot_ic50':         'Half-inhibitory conc.',
        'chart_conc_title':   'Concentration–Time Curve (24 h)',
        'chart_x_time':       'Time (h)',
        'chart_y_conc':       'Concentration (mg/L)',
        'metric_cmax_blood':  'Peak Blood Conc. (Cmax)',
        'metric_cmax_tissue': 'Peak Synovium Conc.',
        'metric_tmax':        'Time to Peak (Tmax)',
        'metric_auc':         'Total Drug Exposure (AUC₀₋₂₄)',

        # Section ③ Dose comparison
        'section3_header':      '③ Dose Comparison',
        'table_col_dose':       'Dose (mg)',
        'table_col_cmax':       'Peak Blood Conc.',
        'table_col_auc':        'Total Exposure',
        'table_col_risk':       'Risk Level',
        'table_current_marker': ' ← current',

        # Section ④ Animation
        'section4_header': '④ Drug Pathway in Your Body',
        'organ_blood':    'Blood',
        'organ_liver':    'Liver',
        'organ_synovium': 'Joint / Synovium',
        'state_safe':   'Safe ✓',
        'state_warn':   'Caution ⚠',
        'state_danger': 'Danger ⚠',

        # Section ⑤ Model explanation
        'section5_header': '⑤ How Does This Tool Work?',
        'expander_label':  'Learn More',
        'expander_content': """\
### How It Works

1. **Physiologically Based Pharmacokinetic Model (PBPK)** — Calculates in real time how the drug travels through intestine → liver → blood → synovium using mathematical equations.
2. **500 Virtual Patients (Monte Carlo)** — Generates training data by simulating patients with varying body weight, kidney function, and enzyme genotype.
3. **Machine Learning Classifier (Random Forest)** — A model achieving 5-fold cross-validated F1-macro ≥ 0.70 predicts the risk category.

### Reading the Animation

- **Red spreading effect** — Drug concentration in that organ has exceeded 50 % of the toxic threshold; side-effect risk detected.
- **Green border + "Safe Pass"** — That organ is within the safe concentration range.
- **Bottom HUD** — Shows ✓ (safe) or ⚠ (risk) for each stage as the drug progresses.

### Glossary

| Term | Definition |
|------|------------|
| Peak Blood Conc. (Cmax) | Maximum drug concentration in blood after dosing |
| Total Drug Exposure (AUC) | Total blood drug exposure integrated over 24 hours |
| Kidney Function (eGFR) | Rate at which kidneys filter blood per minute |
| Enzyme Type (CYP2C9) | Genetic variant of the liver enzyme that metabolizes this drug |
| Half-inhibitory Conc. (IC50) | Minimum concentration needed to reduce inflammation by 50 % |

### Drug Information

| Drug | Standard Dose | Enzyme | IC50 |
|------|--------------|--------|------|
| Ibuprofen | 400 mg | CYP2C9 | 1.0 mg/L |
| Naproxen | 250 mg | CYP2C9 | 0.7 mg/L |
| Celecoxib | 200 mg | CYP2C9 | 0.05 mg/L |
| Acetaminophen | 500 mg | CYP2E1 (not CYP2C9) | 5.0 mg/L |

### Model Limitations

- Protein binding (ibuprofen 99 %) and active metabolites are not explicitly modeled.
- Training data is simulated, not real clinical data.
- Drug–drug interactions, food effects, and formulation differences are not modeled.
""",

        # Disclaimer
        'disclaimer': (
            '⚠️ This tool is an educational simulator for high school research purposes only. '
            'It must not be used for actual clinical decisions. '
            'Always consult a doctor or pharmacist before taking any medication.'
        ),

        # Animation override labels (passed as window.__patientLabels)
        # FocusLabel text pattern: {prefix}{organ_name}{suffix}
        'anim_side_effect_risk':   'Side-Effect Risk',
        'anim_side_effect_prefix': 'Risk detected in ',
        'anim_side_effect_suffix': '',
        'anim_stages':   ['Intake', 'Intestine', 'Liver', 'Blood', 'Joint'],
        'anim_safe_pass': 'Safe Pass',
        'anim_receiving': 'Receiving',
    },
}

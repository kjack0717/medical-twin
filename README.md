# 메디컬 트윈 — COX 저해제 개인 맞춤형 독성 위험 예측

이 프로젝트는 이부프로펜·나프록센·셀레콕시브·아세트아미노펜 4종의 COX 저해제에 대해, PBPK(생리 기반 약동학) ODE 시뮬레이션과 Monte Carlo 가상 환자 생성을 결합하고, 그 결과를 Random Forest/MLP 머신러닝 모델로 학습하여 개인의 신체 조건(체중·신장·신기능·연령 등)에 따른 독성 위험을 예측하는 고등학생 융합 탐구 프로젝트입니다. 최종 결과물은 Streamlit 웹 앱으로 제공됩니다.

---

## 실행 순서

| STEP | 모듈 | 설명 |
|------|------|------|
| 0 | — | 프로젝트 골격 구축 (현재 단계) |
| 1A | `params.py` | PBPK 파라미터 정의 |
| 1B | `data_collection.py` | 공개 DB에서 약물 데이터 수집 |
| 2 | `pbpk_model.py` | PBPK ODE 시뮬레이션 구현 |
| 3 | `virtual_patients.py` | Monte Carlo 가상 환자 500명 생성 |
| 4 | `ml_model.py` | RF/MLP 분류 모델 훈련·평가 |
| 5 | `app.py` | Streamlit 웹 앱 구현 |
| 6 | `run_pipeline.py` | 전체 파이프라인 일괄 실행 |

---

## 설치 및 실행 명령

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 전체 파이프라인 실행 (데이터 수집 → 시뮬레이션 → 모델 훈련)
python run_pipeline.py

# 4. 웹 앱 실행
streamlit run app.py
```

---

## 폴더 구조

```
medical_twin/
├── params.py            # PBPK 파라미터 상수
├── data_collection.py   # 약물 데이터 수집
├── pbpk_model.py        # PBPK ODE 시뮬레이션
├── virtual_patients.py  # Monte Carlo 가상 환자 생성
├── ml_model.py          # RF/MLP 모델 훈련·평가
├── report_generator.py  # PDF 리포트 생성 (fpdf2)
├── egfr_calc.py         # CKD-EPI 2021 eGFR 자동 계산
├── app.py               # Streamlit 웹 앱
├── run_pipeline.py      # 전체 파이프라인 진입점
├── requirements.txt     # 의존성 목록
├── fonts/               # 한글 트루타입 폰트 (PDF 출력용)
├── data/                # 시뮬레이션 결과 CSV 저장 (git 제외)
└── models/              # 학습된 모델 파일 저장
```

---

## PDF 리포트 — 한글 폰트 설치

앱의 "리포트 생성하기" 버튼으로 PDF를 다운로드할 수 있습니다.
한글이 올바르게 출력되려면 `fonts/` 폴더에 **NanumGothic.ttf** 가 있어야 합니다.

```
# 나눔고딕 다운로드 (무료 공개 폰트, SIL OFL 라이선스)
# https://hangeul.naver.com/font 에서 나눔고딕 다운로드 후
# NanumGothic.ttf 를 아래 경로에 복사:
medical_twin/fonts/NanumGothic.ttf
```

- Windows에 나눔고딕이 설치된 경우 자동으로 시스템 폰트를 탐색합니다.
- 폰트가 없으면 영문·숫자만 출력되는 fallback 모드로 동작합니다.
- **Streamlit Cloud 배포 시**: `fonts/NanumGothic.ttf` 를 git에 함께 커밋해야 PDF 한글이 정상 출력됩니다.
- **참고**: `kaleido` 패키지가 없으면 차트 이미지 없이 텍스트만 포함된 PDF가 생성됩니다.

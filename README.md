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
├── app.py               # Streamlit 웹 앱
├── run_pipeline.py      # 전체 파이프라인 진입점
├── requirements.txt     # 의존성 목록
├── data/                # 시뮬레이션 결과 CSV 저장 (git 제외)
└── models/              # 학습된 모델 파일 저장
```

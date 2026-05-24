"""PBPK 시뮬레이션 결과를 학습 데이터로 삼아 RF/MLP 분류 모델을 훈련·평가하는 모듈."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)

# ---------------------------------------------------------------------------
# 한국어 폰트 설정
# ---------------------------------------------------------------------------
def _setup_korean_font() -> None:
    import platform
    _fonts = {'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic', 'Linux': 'NanumGothic'}
    matplotlib.rcParams['font.family'] = _fonts.get(platform.system(), 'sans-serif')
    matplotlib.rcParams['axes.unicode_minus'] = False

_setup_korean_font()

_DATA_DIR  = Path(__file__).parent / "data"
_MODEL_DIR = Path(__file__).parent / "models"

# CYP2C9 순서형 인코딩 — 대사 활성 감소 순서로 정수 배정
_CYP2C9_ORDER: dict[str, int] = {
    '*1/*1': 0, '*1/*2': 1, '*1/*3': 2, '*2/*3': 3, '*3/*3': 4,
}

# 특성 컬럼 순서 (스케일러 저장·로드 시 일치 필요)
_FEATURE_COLS = ['body_weight', 'egfr', 'cyp2c9_enc', 'drug_enc', 'dose_mg']

# 특성 한국어 표시명 (feature importance 플롯용)
_FEATURE_LABELS = ['체중(kg)', 'eGFR', 'CYP2C9\n(순서형)', '약물\n(인코딩)', '용량(mg)']


# =============================================================================
# 전처리
# =============================================================================

def load_and_preprocess(
    csv_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           LabelEncoder, LabelEncoder, StandardScaler]:
    """CSV를 불러와 전처리 후 train/test 분리된 배열을 반환한다.

    전처리 파이프라인:
        1. 'error' 라벨 행 제거
        2. CYP2C9 순서형 인코딩, drug LabelEncoder
        3. train_test_split (stratify=y, test_size=0.2)
        4. StandardScaler fit on train → transform both

    Returns:
        X_train, X_test, y_train, y_test,
        label_encoder, drug_encoder, scaler
    """
    if csv_path is None:
        csv_path = _DATA_DIR / 'patients_dataset.csv'

    df = pd.read_csv(csv_path)
    df = df[df['label'] != 'error'].copy()
    df = df.reset_index(drop=True)

    # 타깃 인코딩 (LabelEncoder는 알파벳 순으로 배정)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    # 약물 LabelEncoder
    drug_encoder = LabelEncoder()
    df['drug_enc'] = drug_encoder.fit_transform(df['drug'])

    # CYP2C9 순서형 인코딩
    df['cyp2c9_enc'] = df['cyp2c9_genotype'].map(_CYP2C9_ORDER).astype(float)

    # 특성 행렬
    X_raw = df[_FEATURE_COLS].values.astype(float)

    # 결측 확인
    if np.isnan(X_raw).any():
        warnings.warn("특성 행렬에 NaN이 있습니다. 해당 행을 제거합니다.")
        mask = ~np.isnan(X_raw).any(axis=1)
        X_raw = X_raw[mask]
        y = y[mask]

    # train / test 분리 (scaler leak 방지를 위해 분리 후 fit)
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42,
    )

    # StandardScaler fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_tr_raw)
    X_test  = scaler.transform(X_te_raw)

    return X_train, X_test, y_train, y_test, label_encoder, drug_encoder, scaler


# =============================================================================
# 모델 정의
# =============================================================================

def build_rf() -> RandomForestClassifier:
    """Random Forest 분류 모델을 생성한다."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )


def build_mlp() -> MLPClassifier:
    """MLP 분류 모델을 생성한다."""
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )


# =============================================================================
# 시각화
# =============================================================================

def plot_feature_importance(
    rf_model: RandomForestClassifier,
    save_path: Path,
) -> None:
    """RF 특성 중요도 막대그래프를 저장한다."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    bars = ax.bar(
        range(len(importances)),
        importances[indices],
        color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'],
        edgecolor='white',
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [_FEATURE_LABELS[i] for i in indices], fontsize=10,
    )
    ax.set_ylabel('특성 중요도 (Gini)', fontsize=11)
    ax.set_title('Random Forest — 특성 중요도', fontsize=13, fontweight='bold')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_ylim(0, importances.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [저장] {save_path.name}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    """혼동 행렬을 히트맵으로 저장한다."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel('예측 라벨', fontsize=11)
    ax.set_ylabel('실제 라벨', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [저장] {save_path.name}")


# =============================================================================
# 모델 선택 논증
# =============================================================================

def compare_and_justify(
    rf_cv_scores: np.ndarray,
    mlp_cv_scores: np.ndarray,
    rf_test_f1: float,
    mlp_test_f1: float,
) -> str:
    """RF와 MLP의 성능을 비교하고 모델 선택 논거를 한국어로 반환한다.

    Args:
        rf_cv_scores:   RF 5-fold CV F1(macro) 배열
        mlp_cv_scores:  MLP 5-fold CV F1(macro) 배열
        rf_test_f1:     RF 테스트 셋 F1(macro)
        mlp_test_f1:    MLP 테스트 셋 F1(macro)

    Returns:
        모델 선택 논증 문자열 (data/model_comparison_note.txt에 저장됨)
    """
    rf_mean,  rf_std  = float(np.mean(rf_cv_scores)),  float(np.std(rf_cv_scores))
    mlp_mean, mlp_std = float(np.mean(mlp_cv_scores)), float(np.std(mlp_cv_scores))

    winner    = 'RF' if rf_test_f1 >= mlp_test_f1 else 'MLP'
    stab_note = (
        f"RF의 CV 표준편차({rf_std:.4f})가 MLP({mlp_std:.4f})보다 "
        + ("낮아 일반화 안정성이 더 높습니다." if rf_std <= mlp_std else "높으나 이는 소규모 불균형 데이터 특성입니다.")
    )

    text = (
        f"[모델 선택 분석: Random Forest vs MLP]\n"
        f"\n"
        f"데이터: 500명 가상환자, 3-클래스 독성 분류 (standard / dose_adjust / toxic)\n"
        f"\n"
        f"[교차검증 F1-macro (5-fold)]\n"
        f"  RF  : {rf_mean:.4f} +/- {rf_std:.4f}  (폴드별: "
        + ", ".join(f"{s:.3f}" for s in rf_cv_scores) + ")\n"
        f"  MLP : {mlp_mean:.4f} +/- {mlp_std:.4f}  (폴드별: "
        + ", ".join(f"{s:.3f}" for s in mlp_cv_scores) + ")\n"
        f"\n"
        f"[테스트 셋 F1-macro]\n"
        f"  RF  = {rf_test_f1:.4f},  MLP = {mlp_test_f1:.4f}\n"
        f"\n"
        f"[모델 선택 논거]\n"
        f"1. 안정성: {stab_note}\n"
        f"2. 해석 가능성: RF는 feature_importances_ 를 제공하여 임상 인자(체중·eGFR·유전형)의\n"
        f"   기여도를 정량화할 수 있습니다. MLP는 블랙박스 구조로 해석이 어렵습니다.\n"
        f"3. 소규모 불균형 데이터: n=500에서 MLP는 hyperparameter 민감도가 높아 과적합\n"
        f"   위험이 있습니다. RF의 앙상블 구조(200 trees)는 이를 완화합니다.\n"
        f"4. 클래스 불균형 처리: RF class_weight='balanced' 설정으로 희소 클래스(toxic, ~4%)\n"
        f"   탐지 성능을 명시적으로 강화했습니다.\n"
        f"\n"
        f"=> 최종 채택 모델: {winner} (테스트 F1 기준)\n"
    )
    return text


# =============================================================================
# 메인 학습 절차
# =============================================================================

def main() -> None:
    """전체 ML 파이프라인을 실행한다."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 전처리
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  STEP 4 - ML 모델 학습 및 평가")
    print("=" * 60)
    print("\n[1] 데이터 로드 및 전처리 ...")
    (X_train, X_test, y_train, y_test,
     label_enc, drug_enc, scaler) = load_and_preprocess()

    label_names = list(label_enc.classes_)  # alphabetical: dose_adjust, standard, toxic
    print(f"  Train: {X_train.shape},  Test: {X_test.shape}")
    print(f"  클래스: {label_names}")
    print(f"  Train 분포: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # -------------------------------------------------------------------------
    # 2. 모델 생성 및 5-fold CV
    # -------------------------------------------------------------------------
    print("\n[2] 5-fold Stratified CV ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf  = build_rf()
    mlp = build_mlp()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf_cv  = cross_val_score(rf,  X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
        mlp_cv = cross_val_score(mlp, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)

    print(f"  RF  CV F1-macro: {np.mean(rf_cv):.4f} +/- {np.std(rf_cv):.4f}  {np.round(rf_cv, 4)}")
    print(f"  MLP CV F1-macro: {np.mean(mlp_cv):.4f} +/- {np.std(mlp_cv):.4f}  {np.round(mlp_cv, 4)}")

    # -------------------------------------------------------------------------
    # 3. 전체 train으로 재학습 → test 평가
    # -------------------------------------------------------------------------
    print("\n[3] 최종 학습 및 테스트 평가 ...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf.fit(X_train, y_train)
        mlp.fit(X_train, y_train)

    y_pred_rf  = rf.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)

    rf_test_f1  = f1_score(y_test, y_pred_rf,  average='macro', zero_division=0)
    mlp_test_f1 = f1_score(y_test, y_pred_mlp, average='macro', zero_division=0)
    rf_acc      = accuracy_score(y_test, y_pred_rf)
    mlp_acc     = accuracy_score(y_test, y_pred_mlp)

    print(f"\n  [RF  테스트]  F1-macro={rf_test_f1:.4f}  Accuracy={rf_acc:.4f}")
    print(classification_report(y_test, y_pred_rf,  target_names=label_names, zero_division=0))
    print(f"\n  [MLP 테스트]  F1-macro={mlp_test_f1:.4f}  Accuracy={mlp_acc:.4f}")
    print(classification_report(y_test, y_pred_mlp, target_names=label_names, zero_division=0))

    # -------------------------------------------------------------------------
    # 4. 시각화
    # -------------------------------------------------------------------------
    print("\n[4] 시각화 저장 ...")
    plot_feature_importance(rf, _DATA_DIR / 'feature_importance.png')
    plot_confusion_matrix(
        y_test, y_pred_rf, label_names,
        'RF 혼동 행렬', _DATA_DIR / 'confusion_matrix_rf.png',
    )
    plot_confusion_matrix(
        y_test, y_pred_mlp, label_names,
        'MLP 혼동 행렬', _DATA_DIR / 'confusion_matrix_mlp.png',
    )

    # -------------------------------------------------------------------------
    # 5. 모델 저장
    # -------------------------------------------------------------------------
    print("\n[5] 모델 저장 ...")
    joblib.dump(rf,        _MODEL_DIR / 'model_rf.pkl')
    joblib.dump(mlp,       _MODEL_DIR / 'model_mlp.pkl')
    joblib.dump(scaler,    _MODEL_DIR / 'feature_scaler.pkl')
    joblib.dump(label_enc, _MODEL_DIR / 'label_encoder.pkl')
    joblib.dump(drug_enc,  _MODEL_DIR / 'drug_encoder.pkl')
    for name in ['model_rf', 'model_mlp', 'feature_scaler', 'label_encoder', 'drug_encoder']:
        print(f"  [저장] models/{name}.pkl")

    # -------------------------------------------------------------------------
    # 6. 비교 분석 및 CSV 저장
    # -------------------------------------------------------------------------
    print("\n[6] 모델 비교 저장 ...")
    note = compare_and_justify(rf_cv, mlp_cv, rf_test_f1, mlp_test_f1)
    print(note)
    note_path = _DATA_DIR / 'model_comparison_note.txt'
    note_path.write_text(note, encoding='utf-8')
    print(f"  [저장] {note_path.name}")

    comparison_df = pd.DataFrame([
        {
            'model':        'RandomForest',
            'cv_f1_mean':   round(float(np.mean(rf_cv)),  4),
            'cv_f1_std':    round(float(np.std(rf_cv)),   4),
            'test_f1':      round(rf_test_f1,  4),
            'test_accuracy': round(rf_acc,      4),
        },
        {
            'model':        'MLP',
            'cv_f1_mean':   round(float(np.mean(mlp_cv)), 4),
            'cv_f1_std':    round(float(np.std(mlp_cv)),  4),
            'test_f1':      round(mlp_test_f1, 4),
            'test_accuracy': round(mlp_acc,     4),
        },
    ])
    comp_csv = _DATA_DIR / 'model_comparison.csv'
    comparison_df.to_csv(comp_csv, index=False, encoding='utf-8')
    print(f"  [저장] {comp_csv.name}")

    # -------------------------------------------------------------------------
    # 7. 자가 검증
    # -------------------------------------------------------------------------
    print("\n[7] 자가 검증 ...")
    assert np.mean(rf_cv) >= 0.70, \
        f"RF CV F1 {np.mean(rf_cv):.4f} < 0.70 — 모델 성능 미달"

    required_models = [
        'model_rf.pkl', 'model_mlp.pkl',
        'feature_scaler.pkl', 'label_encoder.pkl', 'drug_encoder.pkl',
    ]
    for fname in required_models:
        assert (_MODEL_DIR / fname).exists(), f"모델 파일 없음: {fname}"

    for png in ['confusion_matrix_rf.png', 'confusion_matrix_mlp.png']:
        assert (_DATA_DIR / png).exists(), f"PNG 없음: {png}"

    assert comp_csv.exists(), "model_comparison.csv 없음"
    comp_check = pd.read_csv(comp_csv)
    required_cols = {'model', 'cv_f1_mean', 'cv_f1_std', 'test_f1', 'test_accuracy'}
    assert required_cols.issubset(comp_check.columns), \
        f"model_comparison.csv 컬럼 오류: {comp_check.columns.tolist()}"

    print("  모든 검증 통과")
    print("\nSTEP 4 통과")


if __name__ == '__main__':
    main()

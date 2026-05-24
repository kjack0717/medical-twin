"""전체 파이프라인(데이터 수집 → PBPK → 가상환자 → ML 훈련)을 순서대로 실행하는 진입점 모듈."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── 로거 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_DATA_DIR  = _ROOT / "data"
_MODEL_DIR = _ROOT / "models"

# 최종 검증 대상 파일 목록
_REQUIRED_OUTPUTS = [
    'data/drugs_descriptors.csv',
    'data/patients_dataset.csv',
    'data/feature_importance.png',
    'data/model_comparison.csv',
    'models/model_rf.pkl',
    'models/model_mlp.pkl',
]


# =============================================================================
# 단계별 실행 함수
# =============================================================================

def run_data_collection() -> None:
    """STEP 1B: PubChem API로 약물 분자 기술자를 수집하고 CSV로 저장한다."""
    log.info("STEP 1B 시작 — PubChem 데이터 수집")
    try:
        from data_collection import collect_all
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        df = collect_all()
        out = _DATA_DIR / 'drugs_descriptors.csv'
        df.to_csv(out, index=False, encoding='utf-8')
        log.info("STEP 1B 완료 — %s (%d행)", out.name, len(df))
    except Exception as exc:
        log.error("STEP 1B 실패: %s", exc, exc_info=True)
        sys.exit(1)


def run_virtual_patients(n: int = 500) -> None:
    """STEP 3: Monte Carlo 가상환자 n명을 생성하고 CSV로 저장한다."""
    log.info("STEP 3 시작 — 가상환자 %d명 생성 (PBPK 포함)", n)
    try:
        from virtual_patients import generate_dataset
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n=n, seed=42, verbose=False)
        out = _DATA_DIR / 'patients_dataset.csv'
        df.to_csv(out, index=False, encoding='utf-8')
        label_dist = df['label'].value_counts().to_dict()
        log.info("STEP 3 완료 — %s (%d행) 라벨분포: %s", out.name, len(df), label_dist)
        err_rate = (df['label'] == 'error').mean()
        if err_rate >= 0.05:
            log.warning("오류 비율 %.1f%% — 5%% 이상, PBPK 파라미터를 점검하세요.", err_rate * 100)
    except Exception as exc:
        log.error("STEP 3 실패: %s", exc, exc_info=True)
        sys.exit(1)


def run_ml() -> None:
    """STEP 4: RF/MLP 분류 모델을 훈련·평가하고 모델 파일을 저장한다."""
    log.info("STEP 4 시작 — ML 모델 학습")
    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from ml_model import main as ml_main
            ml_main()
        log.info("STEP 4 완료 — models/ 폴더에 pkl 저장됨")
    except Exception as exc:
        log.error("STEP 4 실패: %s", exc, exc_info=True)
        sys.exit(1)


# =============================================================================
# 통합 검증
# =============================================================================

def verify_outputs() -> bool:
    """필수 산출물이 모두 존재하는지 확인하고 결과를 반환한다."""
    all_ok = True
    for rel_path in _REQUIRED_OUTPUTS:
        full = _ROOT / rel_path
        if full.exists():
            size_kb = full.stat().st_size / 1024
            log.info("  [OK] %-40s (%.1f KB)", rel_path, size_kb)
        else:
            log.error("  [누락] %s", rel_path)
            all_ok = False
    return all_ok


# =============================================================================
# CLI 진입점
# =============================================================================

def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description='COX 메디컬 트윈 파이프라인 실행기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '예시:\n'
            '  python run_pipeline.py --steps all\n'
            '  python run_pipeline.py --steps patients,ml --n_patients 200\n'
            '  python run_pipeline.py --skip_data --steps all\n'
        ),
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help="실행할 단계: 'all' 또는 콤마 구분 부분집합 'data,patients,ml' (기본: all)",
    )
    parser.add_argument(
        '--n_patients',
        type=int,
        default=500,
        help='생성할 가상 환자 수 (기본: 500)',
    )
    parser.add_argument(
        '--skip_data',
        action='store_true',
        default=False,
        help='PubChem API 호출(STEP 1B)을 건너뜁니다',
    )
    return parser.parse_args()


def main() -> None:
    """인자를 파싱하고 선택된 단계를 순차적으로 실행한다."""
    args = parse_args()

    # 실행할 단계 결정
    if args.steps.strip().lower() == 'all':
        selected = {'data', 'patients', 'ml'}
    else:
        selected = {s.strip().lower() for s in args.steps.split(',')}
        valid = {'data', 'patients', 'ml'}
        unknown = selected - valid
        if unknown:
            log.error("알 수 없는 단계: %s  (유효값: %s)", unknown, valid)
            sys.exit(1)

    if args.skip_data:
        selected.discard('data')
        log.info("--skip_data 플래그: PubChem 수집(data) 건너뜀")

    log.info("=" * 55)
    log.info("COX 메디컬 트윈 파이프라인 시작")
    log.info("실행 단계: %s  /  가상환자: %d명", sorted(selected), args.n_patients)
    log.info("=" * 55)

    pipeline_start = time.time()
    step_times: dict[str, float] = {}

    # ── STEP 1B: 데이터 수집 ────────────────────────────────────────────────
    if 'data' in selected:
        t0 = time.time()
        run_data_collection()
        step_times['data'] = time.time() - t0
        log.info("STEP 1B 소요 시간: %.1fs", step_times['data'])

    # ── STEP 3: 가상환자 생성 ───────────────────────────────────────────────
    if 'patients' in selected:
        t0 = time.time()
        run_virtual_patients(n=args.n_patients)
        step_times['patients'] = time.time() - t0
        log.info("STEP 3 소요 시간: %.1fs", step_times['patients'])

    # ── STEP 4: ML 학습 ─────────────────────────────────────────────────────
    if 'ml' in selected:
        t0 = time.time()
        run_ml()
        step_times['ml'] = time.time() - t0
        log.info("STEP 4 소요 시간: %.1fs", step_times['ml'])

    # ── 통합 검증 ───────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("산출물 검증 중 ...")
    all_ok = verify_outputs()

    total = time.time() - pipeline_start
    log.info("=" * 55)
    log.info("단계별 소요 시간: %s", {k: f'{v:.1f}s' for k, v in step_times.items()})
    log.info("전체 소요 시간: %.1fs", total)

    if all_ok:
        log.info("전체 파이프라인 통과")
        print("\n전체 파이프라인 통과")
    else:
        log.error("일부 산출물이 누락되었습니다. 위 오류를 확인하세요.")
        sys.exit(1)


if __name__ == '__main__':
    main()

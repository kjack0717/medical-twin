"""
train_classifier.py — MobileNetV2 Transfer Learning 의약품 3분류 모델 학습
대상: 이부프로펜 / 나프록센 / 아세트아미노펜

실행 전 STEP 1(crawl_images.py)을 완료하여 data/raw 폴더를 준비하세요.
"""

import os
import json
import shutil
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── 상수 ────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)          # MobileNetV2 입력 크기
BATCH_SIZE = 16                # 일반 노트북 기준 안전한 배치 크기
EPOCHS = 30                    # 최대 에포크 (EarlyStopping으로 조기 종료)
LEARNING_RATE = 0.0001         # 미세조정 학습률
CONFIDENCE_THRESHOLD = 0.70    # 앱에서 사용할 신뢰도 임계값

CLASS_NAMES = ['acetaminophen', 'ibuprofen', 'naproxen']  # 알파벳 순 정렬 — 반드시 이 순서 유지
CLASS_KOREAN = {                # 앱 표시용 한국어 매핑
    'acetaminophen': '아세트아미노펜',
    'ibuprofen': '이부프로펜',
    'naproxen': '나프록센',
}

DATA_DIR  = 'data/raw'                       # 원본 데이터
TRAIN_DIR = 'data/train'                     # 학습용 (80%)
VAL_DIR   = 'data/val'                       # 검증용 (20%)
MODEL_PATH = 'models/drug_classifier.h5'     # 저장 모델 경로
META_PATH  = 'models/classifier_meta.json'   # 메타데이터 경로


# ── 1. 데이터 분할 ────────────────────────────────────────────────────────────
def prepare_data_split(
    data_dir: str = DATA_DIR,
    train_dir: str = TRAIN_DIR,
    val_dir: str = VAL_DIR,
    val_ratio: float = 0.2,
) -> dict:
    """
    data/raw의 이미지를 train(80%)/val(20%)로 분할하여 복사한다.
    분할은 클래스 내에서 무작위(random_state=42)로 수행.
    반환: {'train': {class: n}, 'val': {class: n}}
    """
    stats = {'train': {}, 'val': {}}

    for cls in CLASS_NAMES:
        src_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(src_dir):
            print(f"  [경고] {src_dir} 폴더 없음 — {cls} 건너뜀")
            stats['train'][cls] = 0
            stats['val'][cls] = 0
            continue

        files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.seed(42)
        random.shuffle(files)

        n_val = max(1, int(len(files) * val_ratio))
        val_files   = files[:n_val]
        train_files = files[n_val:]

        for split, split_files, split_dir in [
            ('train', train_files, train_dir),
            ('val',   val_files,   val_dir),
        ]:
            dst = os.path.join(split_dir, cls)
            os.makedirs(dst, exist_ok=True)
            for fname in split_files:
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst, fname))
            stats[split][cls] = len(split_files)

    # 분할 결과 출력
    print("\n━━ 데이터 분할 결과 ━━")
    print(f"{'클래스':<20} {'학습':>6} {'검증':>6}")
    print("-" * 34)
    for cls in CLASS_NAMES:
        t = stats['train'].get(cls, 0)
        v = stats['val'].get(cls, 0)
        print(f"{cls:<20} {t:>6} {v:>6}")
    print()

    return stats


# ── 2. 모델 구성 ──────────────────────────────────────────────────────────────
def build_model() -> tf.keras.Model:
    """
    MobileNetV2 기반의 3분류 Transfer Learning 모델을 구성한다.
    """
    # 기반 모델 로드 (ImageNet 사전학습, 최상단 레이어 제외)
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False  # 사전학습 가중치 동결 (특성 추출기로만 사용)

    # 분류 헤드 추가
    x = base.output
    x = GlobalAveragePooling2D()(x)   # 공간 차원 축소
    x = Dropout(0.3)(x)               # 과적합 방지
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(3, activation='softmax')(x)  # 3분류

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # 학습 가능 파라미터 수 표시
    model.summary()
    return model


# ── 3. 데이터 제너레이터 ──────────────────────────────────────────────────────
def create_data_generators():
    """
    학습/검증 데이터 제너레이터를 생성한다.
    학습 데이터에 Augmentation(증강) 적용.
    반환: (train_gen, val_gen)
    """
    # 학습용: 증강 적용 (회전, 밝기, 좌우반전, 확대/축소)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # 픽셀값 0~1 정규화
        rotation_range=15,           # 최대 ±15도 회전
        brightness_range=[0.7, 1.3], # 밝기 70~130%
        horizontal_flip=True,        # 좌우 반전
        zoom_range=0.1,              # 10% 확대/축소
        width_shift_range=0.05,      # 수평 이동
        height_shift_range=0.05,     # 수직 이동
    )

    # 검증용: 정규화만 (증강 없음 — 실제 성능 측정 왜곡 방지)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,   # 반드시 명시적 순서 지정
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen


# ── 4. 학습 ──────────────────────────────────────────────────────────────────
def train(model, train_gen, val_gen):
    """
    모델을 학습시킨다.
    EarlyStopping으로 과적합을 방지하고 최고 가중치를 자동 복원한다.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,                # 5 에포크 동안 개선 없으면 조기 종료
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n━━ 학습 시작 ━━")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── 5. 평가 및 메타데이터 저장 ────────────────────────────────────────────────
def evaluate_and_save_meta(model, val_gen) -> None:
    """
    최종 모델 성능을 평가하고 메타데이터를 저장한다.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    val_gen.reset()
    y_pred_prob = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.classes

    print("\n━━ 모델 성능 평가 ━━")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("혼동 행렬:")
    print(confusion_matrix(y_true, y_pred))

    # 전체 검증 정확도
    val_acc = float(np.mean(y_pred == y_true))
    print(f"\n✓ 최종 검증 정확도: {val_acc:.1%}")
    if val_acc < 0.70:
        print("⚠ 정확도 70% 미달. 데이터 추가 수집 또는 파라미터 조정 권장.")

    # 메타데이터 저장 (앱에서 로드하여 사용)
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    meta = {
        'class_names': CLASS_NAMES,
        'class_korean': CLASS_KOREAN,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'val_accuracy': round(val_acc, 4),
        'img_size': list(IMG_SIZE),
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"메타데이터 저장 완료: {META_PATH}")


# ── 메인 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # STEP 1 완료 여부 확인
    if not os.path.isdir(DATA_DIR):
        print("[오류] data/raw 폴더가 없습니다.")
        print("  → STEP 1을 먼저 실행하세요: python crawl_images.py")
        raise SystemExit(1)

    # 클래스별 최소 이미지 수 확인
    MIN_IMAGES = 20
    short_classes = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            short_classes.append((cls, 0))
            continue
        count = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if count < MIN_IMAGES:
            short_classes.append((cls, count))

    if short_classes:
        print(f"\n⚠ 다음 클래스의 이미지가 {MIN_IMAGES}장 미만입니다:")
        for cls, n in short_classes:
            print(f"   {cls}: {n}장")
        answer = input("\n계속 진행하시겠습니까? (y/n): ").strip().lower()
        if answer != 'y':
            print("학습을 취소합니다. crawl_images.py로 데이터를 더 수집하세요.")
            raise SystemExit(0)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("MobileNetV2 Transfer Learning 학습 파이프라인 시작")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 1. 데이터 분할
    prepare_data_split()

    # 2. 모델 구성
    model = build_model()

    # 3. 데이터 제너레이터 생성
    train_gen, val_gen = create_data_generators()

    # 4. 학습
    train(model, train_gen, val_gen)

    # 5. 평가 및 메타데이터 저장
    evaluate_and_save_meta(model, val_gen)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("학습 완료")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  모델 저장 경로 : {MODEL_PATH}")
    print(f"  메타데이터 경로: {META_PATH}")
    print("\n다음 단계: python app.py 에서 '📸 포장지 인식' 탭을 확인하세요")

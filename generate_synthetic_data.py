"""
generate_synthetic_data.py
PIL로 합성 약품 포장지 이미지를 생성합니다.
실제 이미지 수집이 어려울 때 사용하는 대체 데이터셋 생성 스크립트.

각 클래스별 실제 제품의 색상·텍스트 특징을 재현:
  ibuprofen     → 주황/갈색 계열 (부루펜, Advil)
  naproxen      → 파란/청록 계열 (탁센나프, Aleve)
  acetaminophen → 빨간/흰색 계열 (타이레놀, Tylenol)
"""

import os
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/raw")
IMAGES_PER_CLASS = 150          # 클래스당 생성 이미지 수
IMG_W, IMG_H = 400, 300         # 생성 이미지 크기 (224 이상)
RANDOM_SEED = 42

# ── 클래스 설정 ───────────────────────────────────────────────────────────────
CLASS_CONFIGS = {
    "ibuprofen": {
        "kr_names":  ["이부프로펜", "부루펜", "이지엔6이브", "애드빌", "탁센이부"],
        "en_names":  ["Ibuprofen", "IBUPROFEN", "Brufen", "Advil"],
        "dose_text": ["200mg", "400mg", "600mg", "1정 / 200mg"],
        "bg_colors": [
            (255, 165,  0), (210, 105, 30), (255, 140,  0),
            (200,  80, 20), (255, 200, 80), (230, 100,  0),
            (180,  70, 10), (245, 170, 50), (160,  60,  0),
        ],
        "text_colors":   [(255, 255, 255), (255, 240, 200)],
        "accent_colors": [(180, 60, 0), (120, 40, 0), (255, 100, 0)],
    },
    "naproxen": {
        "kr_names":  ["나프록센", "탁센나프", "알리브", "낙센", "낙펜"],
        "en_names":  ["Naproxen", "NAPROXEN", "Aleve", "Naprosyn"],
        "dose_text": ["220mg", "250mg", "500mg", "1정 / 220mg"],
        "bg_colors": [
            (0,  100, 200), ( 30, 144, 255), ( 65, 105, 225),
            (0,   70, 180), (100, 180, 255), ( 10, 120, 210),
            (0,   50, 160), ( 50, 130, 220), ( 20,  90, 190),
        ],
        "text_colors":   [(255, 255, 255), (220, 240, 255)],
        "accent_colors": [(0, 50, 150), (0, 30, 120), (30, 80, 200)],
    },
    "acetaminophen": {
        "kr_names":  ["아세트아미노펜", "타이레놀", "세토펜", "판피린", "챔프"],
        "en_names":  ["Acetaminophen", "ACETAMINOPHEN", "Tylenol", "Paracetamol"],
        "dose_text": ["500mg", "650mg", "325mg", "1정 / 500mg"],
        "bg_colors": [
            (220,  20,  60), (178,  34,  34), (139,   0,   0),
            (200,  30,  50), (240,  60,  80), (160,  10,  30),
            (255,  80, 100), (180,  20,  40), (210,  40,  60),
        ],
        "text_colors":   [(255, 255, 255), (255, 220, 220)],
        "accent_colors": [(120, 0, 0), (80, 0, 0), (200, 0, 0)],
    },
}

# ── 폰트 로드 ─────────────────────────────────────────────────────────────────
def _load_fonts():
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",    # 맑은 고딕 (한국어)
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    found = None
    for p in candidates:
        if Path(p).exists():
            found = p
            break

    def _font(size):
        try:
            return ImageFont.truetype(found, size) if found else ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    return {
        "lg":  _font(42),
        "md":  _font(28),
        "sm":  _font(20),
        "xs":  _font(14),
    }

FONTS = _load_fonts()


# ── 드로잉 헬퍼 ───────────────────────────────────────────────────────────────
def _darken(color, factor=0.7):
    return tuple(int(c * factor) for c in color)

def _lighten(color, factor=1.3):
    return tuple(min(255, int(c * factor)) for c in color)

def _random_pill(draw, x, y, r, color):
    """단순 타원형 알약 아이콘을 그린다."""
    draw.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6],
                 fill=color, outline=_darken(color))
    # 알약 가운데 선
    draw.line([x - r * 0.5, y, x + r * 0.5, y],
              fill=_darken(color, 0.5), width=2)

def _draw_background(draw, w, h, bg_color, accent, variant):
    """배경 스타일을 variant에 따라 그린다."""
    draw.rectangle([0, 0, w, h], fill=bg_color)

    if variant == 0:
        # 상단 띠
        draw.rectangle([0, 0, w, h // 5], fill=accent)
        draw.rectangle([0, h - h // 5, w, h], fill=accent)
    elif variant == 1:
        # 좌측 세로 띠
        draw.rectangle([0, 0, w // 6, h], fill=accent)
    elif variant == 2:
        # 둥근 사각 라벨 느낌 (안쪽 밝은 직사각형)
        margin = 18
        inner = _lighten(bg_color, 1.15)
        draw.rounded_rectangle([margin, margin, w - margin, h - margin],
                                radius=16, fill=inner, outline=accent, width=4)
    elif variant == 3:
        # 대각선 스트라이프 (희미)
        stripe_color = _darken(bg_color, 0.85)
        for i in range(-h, w + h, 28):
            draw.line([(i, 0), (i + h, h)], fill=stripe_color, width=6)
    else:
        # 단색 + 테두리
        draw.rectangle([8, 8, w - 8, h - 8],
                        outline=accent, width=5)


def _draw_pill_pattern(draw, w, h, accent, n=5):
    """배경에 희미한 알약 패턴 추가."""
    pale = tuple(min(255, int(c * 1.25)) for c in accent)
    for _ in range(n):
        px = random.randint(20, w - 20)
        py = random.randint(20, h - 20)
        pr = random.randint(12, 22)
        _random_pill(draw, px, py, pr, pale)


def _center_text(draw, y, text, font, fill, w, shadow=True):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (w - tw) // 2
    if shadow:
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 80))
    draw.text((x, y), text, font=font, fill=fill)
    return bbox[3] - bbox[1]  # text height


# ── 이미지 한 장 생성 ──────────────────────────────────────────────────────────
def make_image(class_name: str, cfg: dict, idx: int) -> Image.Image:
    rng = random.Random(RANDOM_SEED + idx * 7)

    w, h = IMG_W, IMG_H
    bg    = rng.choice(cfg["bg_colors"])
    tc    = rng.choice(cfg["text_colors"])
    acc   = rng.choice(cfg["accent_colors"])
    var   = idx % 5  # 배경 스타일 순환

    img  = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    _draw_background(draw, w, h, bg, acc, var)
    _draw_pill_pattern(draw, w, h, acc, n=rng.randint(2, 6))

    # ── 텍스트 레이아웃 ──────────────────────────────────────────────────────
    kr_name  = rng.choice(cfg["kr_names"])
    en_name  = rng.choice(cfg["en_names"])
    dose     = rng.choice(cfg["dose_text"])

    y_cursor = 40
    th = _center_text(draw, y_cursor, kr_name,  FONTS["lg"], tc, w)
    y_cursor += th + 14
    th = _center_text(draw, y_cursor, en_name,  FONTS["md"], tc, w, shadow=False)
    y_cursor += th + 10
    th = _center_text(draw, y_cursor, dose,     FONTS["sm"], acc if var < 3 else tc, w, shadow=False)
    y_cursor += th + 12

    # 구분선
    draw.line([(30, y_cursor), (w - 30, y_cursor)], fill=tc, width=2)
    y_cursor += 10

    # 하단 용도 텍스트
    usage_lines = ["해열 · 진통 · 소염", "Analgesic / Antipyretic"]
    for line in usage_lines:
        _center_text(draw, y_cursor, line, FONTS["xs"],
                     _lighten(tc, 0.85), w, shadow=False)
        y_cursor += 18

    # ── 가벼운 노이즈 추가 (실사진처럼) ──────────────────────────────────────
    if rng.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # ── 랜덤 회전 (±10도) ────────────────────────────────────────────────────
    angle = rng.uniform(-10, 10)
    if abs(angle) > 1.5:
        img = img.rotate(angle, expand=False, fillcolor=bg)

    return img


# ── 메인 ─────────────────────────────────────────────────────────────────────
def generate_all():
    random.seed(RANDOM_SEED)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("합성 약품 포장지 데이터셋 생성")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for cls, cfg in CLASS_CONFIGS.items():
        out_dir = OUTPUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{cls}] {IMAGES_PER_CLASS}장 생성 중...")
        for i in range(IMAGES_PER_CLASS):
            img = make_image(cls, cfg, i)
            fname = out_dir / f"{cls}_synth_{i:04d}.jpg"
            img.save(str(fname), "JPEG", quality=92)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{IMAGES_PER_CLASS} 완료")

        count = len(list(out_dir.glob("*.jpg")))
        print(f"  [{cls}] 저장 완료: {count}장 → {out_dir}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    total = sum(len(list((OUTPUT_DIR / c).glob("*.jpg"))) for c in CLASS_CONFIGS)
    print(f"전체 생성 완료: {total}장")
    print("다음 단계: python train_classifier.py")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    generate_all()

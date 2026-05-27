"""PDF 리포트 생성 모듈 (fpdf2 기반).

한글 출력을 위해 NanumGothic.ttf 또는 시스템 한글 폰트를 사용한다.
폰트가 없을 경우 영문/숫자만 출력되는 fallback 모드로 동작한다.
"""

import io
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ── 폰트 경로 ─────────────────────────────────────────────────────────────────
_FONT_DIR  = Path(__file__).parent / 'fonts'
_FONT_PATH = _FONT_DIR / 'NanumGothic.ttf'   # 프로젝트 동봉 폰트 (1순위)

# ── 위험 등급별 레이블 / 텍스트 색(RGB) / 배경 색(RGB) ───────────────────────
_LABEL_KO = {
    'standard':    ('안전 (표준)',      (46, 125,  50), (232, 245, 233)),
    'dose_adjust': ('주의 (용량 조정)', (230,  81,   0), (255, 243, 224)),
    'toxic':       ('위험 (독성)',      (183,  28,  28), (255, 235, 238)),
}
_LABEL_EN = {
    'standard':    ('Safe (Standard)',      (46, 125,  50), (232, 245, 233)),
    'dose_adjust': ('Caution (Dose Adj.)', (230,  81,   0), (255, 243, 224)),
    'toxic':       ('Danger (Toxic)',       (183,  28,  28), (255, 235, 238)),
}

# 환자 정보 테이블 키(한국어 → 영어) 매핑 — fallback 모드용
_KEY_KO_TO_EN: dict[str, str] = {
    '약물':         'Drug',
    '복용량':       'Dose',
    '체중':         'Weight',
    '나이':         'Age',
    'eGFR':         'eGFR',
    'CYP2C9 유전형': 'CYP2C9',
}


# =============================================================================
# 한글 폰트 탐색
# =============================================================================

def _resolve_korean_font() -> Path | None:
    """사용 가능한 한글 트루타입 폰트 경로를 반환. 없으면 None.

    탐색 순서:
        1. 프로젝트 fonts/NanumGothic.ttf
        2. OS별 기본 한글 폰트 (Malgun Gothic / AppleGothic / NanumGothic)
    """
    # 1순위: 프로젝트 동봉 폰트
    if _FONT_PATH.exists():
        return _FONT_PATH

    # 2순위: 시스템 기본 한글 폰트
    import platform
    _sys_candidates: dict[str, list[Path]] = {
        'Windows': [
            Path(r'C:\Windows\Fonts\malgun.ttf'),       # Malgun Gothic (Windows 기본)
            Path(r'C:\Windows\Fonts\NanumGothic.ttf'),
        ],
        'Darwin':  [Path('/Library/Fonts/AppleGothic.ttf')],
        'Linux':   [Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')],
    }
    for p in _sys_candidates.get(platform.system(), []):
        if p.exists():
            return p
    return None


# =============================================================================
# PDF 클래스 (헤더·푸터 없음)
# =============================================================================

class _MedTwinPDF(FPDF):
    """헤더·푸터를 사용하지 않는 A4 PDF 클래스."""
    def header(self): pass
    def footer(self): pass


# =============================================================================
# PDF 리포트 생성 함수
# =============================================================================

def generate_pdf_report(
    patient_info: dict,
    sim: dict,
    label: str,
    drug_korean: str,
    chart_png_bytes: bytes | None,
) -> bytes:
    """환자 정보와 시뮬레이션 결과로 PDF 리포트를 생성하여 bytes로 반환한다.

    Args:
        patient_info     : 입력 정보 딕셔너리 (키는 한국어 또는 영어)
        sim              : simulate_pbpk() 반환 딕셔너리
        label            : 위험 등급 ('standard' / 'dose_adjust' / 'toxic')
        drug_korean      : 약물 한국어 이름 (제목 표시용)
        chart_png_bytes  : 농도-시간 곡선 PNG bytes (None이면 차트 생략)

    Returns:
        PDF 파일 bytes
    """
    font_path  = _resolve_korean_font()
    use_korean = font_path is not None

    # ── 레이블·메트릭 텍스트 선택 ──────────────────────────────────────────
    _lmap = _LABEL_KO if use_korean else _LABEL_EN
    label_text, text_rgb, bg_rgb = _lmap.get(label, _lmap['standard'])

    # fallback 모드: 환자 정보 키를 영어로 변환, 한글 값은 ASCII 부분만 유지
    if not use_korean:
        patient_info = {
            _KEY_KO_TO_EN.get(k, k): _strip_hangul(str(v))
            for k, v in patient_info.items()
        }
        drug_korean = _strip_hangul(drug_korean)

    # ── PDF 초기화 ─────────────────────────────────────────────────────────
    pdf = _MedTwinPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── 폰트 등록 ──────────────────────────────────────────────────────────
    if use_korean:
        # fpdf2 2.x: Unicode 기본 지원, 같은 파일로 bold 스타일 대체
        pdf.add_font('KO', style='',  fname=str(font_path))
        pdf.add_font('KO', style='B', fname=str(font_path))
        F_REG  = 'KO'
        F_BOLD = 'KO'
    else:
        F_REG  = 'Helvetica'
        F_BOLD = 'Helvetica'

    # ── 레이아웃 상수 ──────────────────────────────────────────────────────
    LM = 15          # 왼쪽 여백 (mm)
    RM = 15          # 오른쪽 여백 (mm)
    PW = 210         # A4 폭 (mm)
    CW = PW - LM - RM   # 콘텐츠 폭 = 180mm
    RH = 7           # 기본 행 높이 (mm)
    NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # 줄바꿈 파라미터

    def _font(style: str, size: int):
        fname = F_BOLD if style == 'B' else F_REG
        pdf.set_font(fname, style=style, size=size)

    # ======================================================================
    # 제목 & 날짜
    # ======================================================================
    _font('B', 18)
    title = '나의 약물 메디컬 트윈 리포트' if use_korean else 'My Drug Medical Twin Report'
    pdf.set_xy(LM, 15)
    pdf.cell(CW, 10, title, align='C', **NL)

    _font('', 9)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    gen_label = '생성 일시' if use_korean else 'Generated'
    pdf.cell(CW, 5, f'{gen_label}: {now_str}', align='C', **NL)

    # 구분선
    y_sep = pdf.get_y() + 2
    pdf.set_draw_color(180, 180, 180)
    pdf.line(LM, y_sep, PW - RM, y_sep)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_y(y_sep + 4)

    # ======================================================================
    # 섹션 1: 입력 정보 표
    # ======================================================================
    _font('B', 11)
    sec1 = '[입력 정보]' if use_korean else '[Patient Information]'
    pdf.cell(CW, 7, sec1, **NL)

    col_key = 65    # 레이블 열 폭
    col_val = CW - col_key

    for key, val in patient_info.items():
        # 레이블 셀 (연한 회색 배경)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_draw_color(200, 200, 200)
        _font('B', 9)
        pdf.cell(col_key, RH, str(key), border=1, fill=True)

        # 값 셀
        pdf.set_fill_color(255, 255, 255)
        _font('', 9)
        pdf.cell(col_val, RH, str(val), border=1, fill=True, **NL)

    pdf.ln(5)
    pdf.set_draw_color(0, 0, 0)

    # ======================================================================
    # 섹션 2: 위험 등급 색상 박스
    # ======================================================================
    _font('B', 11)
    sec2 = '[위험 등급]' if use_korean else '[Risk Level]'
    pdf.cell(CW, 7, sec2, **NL)

    box_x = LM
    box_y = pdf.get_y()
    box_h = 14

    # 배경·테두리 박스 (색상은 등급에 따라 결정)
    pdf.set_fill_color(*bg_rgb)
    pdf.set_draw_color(*text_rgb)
    pdf.set_line_width(0.8)
    pdf.rect(box_x, box_y, CW, box_h, style='FD')
    pdf.set_line_width(0.2)

    # 등급 텍스트 (박스 중앙)
    _font('B', 13)
    pdf.set_text_color(*text_rgb)
    pdf.set_xy(box_x, box_y + 2)
    pdf.cell(CW, 10, label_text, align='C')

    # 색상·텍스트 리셋
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_y(box_y + box_h + 5)

    # ======================================================================
    # 섹션 3: 핵심 PK 지표 표
    # ======================================================================
    _font('B', 11)
    sec3 = '[핵심 PK 지표]' if use_korean else '[Key PK Metrics]'
    pdf.cell(CW, 7, sec3, **NL)

    if use_korean:
        metrics = [
            ('최고 혈중 농도 (Cmax)',      f"{sim['Cmax_blood']:.3f} mg/L"),
            ('최고 활막 농도',             f"{sim['Cmax_tissue']:.3f} mg/L"),
            ('최고 농도 도달 시간 (Tmax)', f"{sim['Tmax_blood']:.2f} h"),
            ('총 노출량 (AUC)',            f"{sim['AUC_blood']:.1f} mg·h/L"),
        ]
    else:
        metrics = [
            ('Peak Blood Conc. (Cmax)', f"{sim['Cmax_blood']:.3f} mg/L"),
            ('Peak Tissue Conc.',       f"{sim['Cmax_tissue']:.3f} mg/L"),
            ('Time to Cmax (Tmax)',     f"{sim['Tmax_blood']:.2f} h"),
            ('Total Exposure (AUC)',    f"{sim['AUC_blood']:.1f} mg·h/L"),
        ]

    col_m_key = 100
    col_m_val = CW - col_m_key

    for m_key, m_val in metrics:
        pdf.set_fill_color(240, 240, 240)
        pdf.set_draw_color(200, 200, 200)
        _font('B', 9)
        pdf.cell(col_m_key, RH, m_key, border=1, fill=True)

        pdf.set_fill_color(255, 255, 255)
        _font('', 9)
        pdf.cell(col_m_val, RH, m_val, border=1, fill=True, **NL)

    pdf.ln(5)
    pdf.set_draw_color(0, 0, 0)

    # ======================================================================
    # 섹션 4: 농도-시간 곡선 이미지
    # ======================================================================
    if chart_png_bytes:
        _font('B', 11)
        sec4 = '[농도-시간 곡선]' if use_korean else '[Concentration-Time Curve]'
        pdf.cell(CW, 7, sec4, **NL)

        # BytesIO로 래핑하여 임시 파일 없이 삽입
        img_buf = io.BytesIO(chart_png_bytes)
        # 가로 폭 CW에 맞추고, 높이는 비율에 따라 자동 (h=0)
        pdf.image(img_buf, x=LM, w=CW, h=0)
        pdf.ln(3)

    # ======================================================================
    # 면책 문구
    # ======================================================================
    pdf.ln(4)
    _font('', 7)
    pdf.set_text_color(120, 120, 120)
    disclaimer = (
        '본 리포트는 교육용 시뮬레이션 결과이며 실제 의료 판단에 사용할 수 없습니다. '
        '복약 전 반드시 의사·약사와 상담하세요.'
        if use_korean else
        'This report is for educational simulation only and must not be used '
        'for actual medical decisions. Always consult a physician or pharmacist.'
    )
    # multi_cell로 자동 줄바꿈
    pdf.multi_cell(CW, 5, disclaimer, align='C')
    pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())


# =============================================================================
# 내부 유틸
# =============================================================================

def _strip_hangul(text: str) -> str:
    """한글을 제거하고 ASCII 범위 문자만 반환 (fallback용)."""
    return ''.join(c for c in text if ord(c) < 0x100).strip()


# =============================================================================
# 자가 검증
# =============================================================================

if __name__ == '__main__':
    font = _resolve_korean_font()
    print(f'한글 폰트: {font}')

    _sim = {
        'Cmax_blood': 12.345, 'Cmax_tissue': 3.210,
        'Tmax_blood': 1.50,   'AUC_blood':  88.7,
    }
    _info = {
        '약물': '이부프로펜 (Ibuprofen)',
        '복용량': '400 mg',
        '체중': '70 kg',
        '나이': '40 세',
        'eGFR': '90 mL/min/1.73m²',
        'CYP2C9 유전형': '*1/*1',
    }

    pdf_bytes = generate_pdf_report(
        patient_info=_info,
        sim=_sim,
        label='dose_adjust',
        drug_korean='이부프로펜',
        chart_png_bytes=None,
    )
    out = Path('data/test_report.pdf')
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(pdf_bytes)
    print(f'PDF 생성 완료: {out}  ({len(pdf_bytes):,} bytes)')

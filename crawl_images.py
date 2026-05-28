"""
crawl_images.py — 식약처 공공데이터 API를 이용한 의약품 이미지 크롤링
대상: 이부프로펜 / 나프록센 / 아세트아미노펜

실행 전 아래 API_KEY를 본인이 발급받은 키로 교체하세요.
발급: https://www.data.go.kr/data/15075057/openapi.do
(식품의약품안전처_의약품개요정보(e약은요) — 무료, 즉시 발급)
"""

import os
import time
import requests
from pathlib import Path

# ── API 키 설정 ──────────────────────────────────────────────────────────────
# TODO: 실행 전 아래 값을 data.go.kr에서 발급받은 실제 API 키로 교체하세요.
API_KEY = "e3beff4e5926f9183034e5b43ff9b527a12c95baef947da18acdbd1af9aa4d15"

BASE_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"

DRUG_KEYWORDS = {
    "ibuprofen": ["이부프로펜", "부루펜", "애드빌", "이지엔6이브", "탁센이부"],
    "naproxen": ["나프록센", "탁센나프", "알리브"],
    "acetaminophen": ["아세트아미노펜", "타이레놀", "세토펜", "판피린"],
}

CLASS_KR = {
    "ibuprofen": "이부프로펜",
    "naproxen": "나프록센",
    "acetaminophen": "아세트아미노펜",
}


def fetch_drug_image_urls(keyword: str, max_images: int = 80) -> list:
    """
    식약처 API로 keyword에 해당하는 의약품의 이미지 URL 목록을 수집한다.
    페이지네이션을 처리하여 max_images개까지 수집.
    반환: 유효한 이미지 URL 리스트 (None 제외)
    """
    urls = []
    num_of_rows = 100
    page_no = 1

    while len(urls) < max_images:
        params = {
            "serviceKey": API_KEY,
            "itemName": keyword,
            "type": "json",
            "numOfRows": num_of_rows,
            "pageNo": page_no,
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
        except requests.RequestException as e:
            print(f"  [오류] 네트워크 요청 실패: {e}")
            break

        if resp.status_code != 200:
            print(f"  [오류] HTTP {resp.status_code} — {keyword} 페이지 {page_no}")
            break

        try:
            data = resp.json()
        except ValueError:
            print(f"  [오류] JSON 파싱 실패 — {keyword} 페이지 {page_no}")
            break

        body = data.get("body", {})
        total_count = body.get("totalCount", 0)
        items = body.get("items", [])

        if not items:
            break

        for item in items:
            img_url = item.get("itemImage")
            if img_url and img_url.strip():
                urls.append(img_url.strip())
            if len(urls) >= max_images:
                break

        # 다음 페이지가 있는지 확인
        if page_no * num_of_rows >= total_count:
            break

        page_no += 1
        time.sleep(0.3)

    print(f"  [{keyword}] {len(urls)}개 URL 수집 완료")
    return urls


def download_image(url: str, save_path: str) -> bool:
    """
    url의 이미지를 다운로드하여 save_path에 저장한다.
    반환: 성공 True, 실패 False
    """
    headers = {"User-Agent": "Mozilla/5.0 (Educational Research Project)"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as e:
        print(f"  [다운로드 오류] {url}: {e}")
        return False

    if resp.status_code != 200:
        return False

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        return False

    if len(resp.content) < 5 * 1024:  # 5KB 미만
        return False

    try:
        with open(save_path, "wb") as f:
            f.write(resp.content)
    except IOError as e:
        print(f"  [저장 오류] {save_path}: {e}")
        return False

    time.sleep(0.5)
    return True


def crawl_drug_class(class_name: str, keywords: list, max_per_keyword: int = 40) -> int:
    """
    하나의 약물 클래스에 대해 여러 키워드로 크롤링하여 이미지를 저장한다.
    반환: 저장된 이미지 총 개수
    """
    save_dir = Path("data") / "raw" / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    seen_urls = set()
    total_saved = 0

    for keyword in keywords:
        print(f"\n━━ [{class_name}] 키워드: {keyword} ━━")

        urls = fetch_drug_image_urls(keyword, max_images=max_per_keyword)
        new_urls = [u for u in urls if u not in seen_urls]
        seen_urls.update(new_urls)

        for idx, url in enumerate(new_urls, start=1):
            filename = f"{class_name}_{keyword}_{idx:04d}.jpg"
            save_path = save_dir / filename

            if save_path.exists():
                total_saved += 1
                continue

            print(f"  {idx}/{len(new_urls)} 다운로드 중...")
            success = download_image(url, str(save_path))
            if success:
                total_saved += 1

        time.sleep(1.0)

    print(f"\n[{class_name}] 총 {total_saved}장 저장 완료 → {save_dir}")
    return total_saved


def verify_dataset(data_dir: str = "data/raw") -> dict:
    """
    다운로드된 데이터셋을 검증하고 통계를 출력한다.
    반환: {class_name: count} 딕셔너리
    """
    from PIL import Image, UnidentifiedImageError

    MIN_COUNT = 30
    results = {}
    base = Path(data_dir)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("데이터셋 검증 결과")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for class_name in DRUG_KEYWORDS:
        class_dir = base / class_name
        if not class_dir.exists():
            print(f"  [{class_name}] 폴더 없음 — 0장")
            results[class_name] = 0
            continue

        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        valid_count = 0
        removed = 0

        for img_path in image_files:
            try:
                with Image.open(img_path) as im:
                    im.verify()
                valid_count += 1
            except (UnidentifiedImageError, Exception):
                print(f"  손상된 파일 제거: {img_path.name}")
                img_path.unlink(missing_ok=True)
                removed += 1

        results[class_name] = valid_count
        kr_name = CLASS_KR.get(class_name, class_name)
        status = "✓" if valid_count >= MIN_COUNT else "⚠ 부족"
        print(f"  {kr_name:16s}: {valid_count:3d}장  {status}")
        if removed:
            print(f"    → 손상 파일 {removed}개 제거됨")
        if valid_count < MIN_COUNT:
            print(f"    → 최소 기준 {MIN_COUNT}장 미달 — keywords에 키워드 추가 후 재실행 권장")

    total = sum(results.values())
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  전체 검증 통과 이미지: {total}장")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return results


# ── 메인 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if API_KEY == "YOUR_KEY_HERE":
        print("[오류] API_KEY가 설정되지 않았습니다.")
        print("  → crawl_images.py 상단의 API_KEY 값을 실제 발급 키로 교체하세요.")
        print("  → 발급: https://www.data.go.kr/data/15075057/openapi.do")
        raise SystemExit(1)

    start_time = time.time()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("의약품 이미지 크롤링 시작")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    counts = {}
    for class_name, keywords in DRUG_KEYWORDS.items():
        counts[class_name] = crawl_drug_class(
            class_name=class_name,
            keywords=keywords,
            max_per_keyword=40,
        )

    stats = verify_dataset("data/raw")

    elapsed = time.time() - start_time
    print(f"\n총 소요 시간: {elapsed:.1f}초")

    print("\n━━ 크롤링 완료 ━━")
    print(f"이부프로펜: {stats.get('ibuprofen', 0)}장")
    print(f"나프록센: {stats.get('naproxen', 0)}장")
    print(f"아세트아미노펜: {stats.get('acetaminophen', 0)}장")
    print("\nSTEP 2로 진행하려면: python train_classifier.py 실행")

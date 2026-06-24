"""임용 공지·교육과정 소식 수집기.

공식 사이트의 HTML 구조가 자주 바뀌고 일부는 동적 렌더링을 사용하므로,
이 모듈은 특정 CSS 선택자에 강하게 묶이지 않는 보수적 수집기로 설계했다.
수집 실패는 서비스 중단이 아니라 해당 출처의 warning으로 기록한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


PROJECT3_DIR = Path(__file__).resolve().parent
SOURCES_PATH = PROJECT3_DIR / "sources.json"
DEFAULT_TIMEOUT = 12
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


@dataclass
class NoticeItem:
    source: str
    category: str
    title: str
    url: str
    matched_keywords: list[str]
    collected_at: str


def load_sources(include_regions: bool = True) -> list[dict]:
    data = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    sources = list(data.get("core_sources", []))
    if include_regions:
        for row in data.get("regional_sources", []):
            sources.append({**row, "category": "시도교육청 공고"})
    return sources


def collect_notices(
    include_regions: bool = True,
    max_per_source: int = 8,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """공식 사이트에서 임용/교육과정 관련 링크 후보를 수집한다."""
    items: list[NoticeItem] = []
    warnings: list[str] = []
    for source in load_sources(include_regions=include_regions):
        try:
            items.extend(_collect_from_source(source, max_per_source=max_per_source, timeout=timeout))
        except Exception as exc:
            warnings.append(f"{source['name']}: {exc}")
    unique_items = _dedupe(items)
    return {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(unique_items),
        "items": [asdict(item) for item in unique_items],
        "warnings": warnings,
        "source_count": len(load_sources(include_regions=include_regions)),
    }


def _collect_from_source(source: dict, max_per_source: int, timeout: int) -> list[NoticeItem]:
    response = requests.get(
        source["url"],
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    soup = BeautifulSoup(response.text, "html.parser")
    keywords = source.get("keywords", [])
    collected: list[NoticeItem] = []
    for anchor in soup.find_all("a"):
        title = _clean_text(anchor.get_text(" ", strip=True))
        if not title or len(title) < 4:
            continue
        matched = _matched_keywords(title, keywords)
        if not matched:
            continue
        href = anchor.get("href") or source["url"]
        if href.strip().lower().startswith("javascript:"):
            continue
        collected.append(
            NoticeItem(
                source=source["name"],
                category=source.get("category", ""),
                title=title[:220],
                url=urljoin(source["url"], href),
                matched_keywords=matched,
                collected_at=datetime.now().isoformat(timespec="seconds"),
            )
        )
        if len(collected) >= max_per_source:
            break
    return collected


def _matched_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    compact = text.replace(" ", "")
    matched = []
    for keyword in keywords:
        if keyword in text or keyword.replace(" ", "") in compact:
            matched.append(keyword)
    return matched


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(items: list[NoticeItem]) -> list[NoticeItem]:
    seen = set()
    output = []
    for item in items:
        key = (item.source, item.title, item.url)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


if __name__ == "__main__":
    print(json.dumps(collect_notices(include_regions=False), ensure_ascii=False, indent=2))

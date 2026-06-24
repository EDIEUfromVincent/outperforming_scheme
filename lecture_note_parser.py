"""인강 기록/강의 요약본 파서.

사용자가 올리는 강의 요약본은 시험 직결 학습자료이므로 일반 참고문서가 아니라
`lecture_note`로 분류해 교과, 주제, 지식 포인트, 기출 언급, 과제를 메타데이터화한다.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


SUBJECT_KEYWORDS = {
    "국어": ["국어", "읽기", "쓰기", "문법", "문학", "화법"],
    "수학": ["수학", "분수", "소수", "도형", "측정", "확률", "통계"],
    "사회": ["사회", "역사", "지리", "민주", "경제", "정치"],
    "과학": ["과학", "탐구", "물질", "생명", "지구", "우주", "전기", "소리"],
    "음악": ["음악", "당김음", "화음", "장조", "단조", "오스티나토", "리코더", "노래"],
    "미술": ["미술", "표현", "감상", "조형"],
    "체육": ["체육", "운동", "스포츠", "건강"],
    "영어": ["영어", "듣기", "말하기", "파닉스"],
    "도덕": ["도덕", "가치", "덕목"],
    "실과": ["실과", "가정", "기술", "소프트웨어"],
    "통합교과": ["바른 생활", "슬기로운 생활", "즐거운 생활", "통합교과"],
}


@dataclass
class LectureNote:
    document_id: str
    title: str
    lecture_date: str | None
    subject: str | None
    topics: list[str]
    exam_years: list[int]
    knowledge_points: list[str]
    assignments: list[str]
    raw_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_lecture_note(text: str, filename: str = "") -> LectureNote:
    normalized = _clean_text(text)
    title = _extract_title(normalized, filename)
    lecture_date = _extract_date(normalized)
    subject = _infer_subject(f"{title}\n{normalized}")
    topics = _extract_topics(normalized)
    exam_years = sorted({int(year) for year in re.findall(r"(20\d{2}|19\d{2})년", normalized)})
    knowledge_points = _extract_numbered_section(normalized, "지식 포인트", ["질문", "과제"])
    assignments = _extract_bullets_after_heading(normalized, "과제")
    document_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return LectureNote(
        document_id=document_id,
        title=title,
        lecture_date=lecture_date,
        subject=subject,
        topics=topics,
        exam_years=exam_years,
        knowledge_points=knowledge_points,
        assignments=assignments,
        raw_text=normalized,
    )


def looks_like_lecture_note(text: str, filename: str = "") -> bool:
    sample = f"{filename}\n{text[:2000]}"
    signals = ["강의", "요약", "지식 포인트", "기출문제", "과제"]
    return sum(1 for signal in signals if signal in sample) >= 2


def _extract_title(text: str, filename: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:180]
    return Path(filename).stem or "강의 요약본"


def _extract_date(text: str) -> str | None:
    match = re.search(r"일시\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}(?:\s+[0-9:]{5,8})?)", text)
    if not match:
        return None
    value = match.group(1).strip()
    try:
        return datetime.fromisoformat(value).isoformat(timespec="seconds")
    except ValueError:
        return value


def _infer_subject(text: str) -> str | None:
    scores = {
        subject: sum(text.count(keyword) for keyword in keywords)
        for subject, keywords in SUBJECT_KEYWORDS.items()
    }
    subject, score = max(scores.items(), key=lambda item: item[1])
    return subject if score > 0 else None


def _extract_topics(text: str) -> list[str]:
    candidates = []
    for pattern in [
        r"음악 이론\s*-\s*([^\n]+)",
        r"강의\s*:\s*([^\n]+)",
        r"요약\s*\n([^\n]+)",
    ]:
        match = re.search(pattern, text)
        if match:
            candidates.extend(re.split(r"[,·/、]| - ", match.group(1)))
    keywords = [
        "당김음", "화음", "장조", "단조", "돌림노래", "짝노래", "메아리노래",
        "오스티나토", "프레이징", "리코더", "나란한조", "주요 3화음",
    ]
    candidates.extend(keyword for keyword in keywords if keyword in text)
    output = []
    seen = set()
    for item in candidates:
        topic = item.strip(" .:-\t")
        if topic.startswith(("이 강의", "강의는", "요약")):
            continue
        topic = re.sub(r"-Summary$", "", topic).strip()
        topic = re.sub(r"(을|를|에 대해|에 관한)?\s*다룬다$", "", topic).strip()
        if len(topic) < 2 or topic in seen:
            continue
        seen.add(topic)
        output.append(topic[:40])
    return output[:20]


def _extract_numbered_section(text: str, heading: str, stop_headings: list[str]) -> list[str]:
    section = _section_text(text, heading, stop_headings)
    if not section:
        return []
    matches = re.findall(r"(?:^|\n)\s*\d+\.\s+([^\n]+)", section)
    if matches:
        return [_clean_inline(item) for item in matches[:30]]
    return [
        _clean_inline(line)
        for line in section.splitlines()
        if line.strip() and len(line.strip()) > 4
    ][:30]


def _extract_bullets_after_heading(text: str, heading: str) -> list[str]:
    section = _section_text(text, heading, [])
    if not section:
        return []
    output = []
    for line in section.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            cleaned = re.sub(r"^[-*]\s+", "", stripped)
            cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
            output.append(_clean_inline(cleaned))
    return output[:30]


def _section_text(text: str, heading: str, stop_headings: list[str]) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    start += len(heading)
    end = len(text)
    for stop in stop_headings:
        idx = text.find(stop, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end].strip()


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()


def _clean_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

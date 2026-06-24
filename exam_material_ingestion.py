"""Utilities for ingesting teacher-exam mock test folders."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any


SUPPORTED_MATERIAL_EXTENSIONS = {".pdf"}
DEFAULT_MOCK_EXAM_DIR = Path("/Users/vincent/Downloads/임용-모의고사")

SUBJECT_KEYWORDS = {
    "국어": "국어",
    "수학": "수학",
    "사회": "사회",
    "과학": "과학",
    "도덕": "도덕",
    "실과": "실과",
    "체육": "체육",
    "음악": "음악",
    "미술": "미술",
    "영어": "영어",
    "통합": "통합교과",
    "총론": "총론",
    "창체": "창의적 체험활동",
    "창의적": "창의적 체험활동",
    "논술": "교직논술",
    "교육과정": "교육과정",
}


def list_mock_exam_pdfs(root: str | Path = DEFAULT_MOCK_EXAM_DIR) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists():
        return []
    return sorted(
        path
        for path in base.rglob("*.pdf")
        if path.is_file() and not path.name.startswith(".")
    )


def infer_mock_exam_metadata(path: str | Path, root: str | Path = DEFAULT_MOCK_EXAM_DIR) -> dict[str, Any]:
    file_path = Path(path)
    base = Path(root).expanduser()
    try:
        relative = file_path.relative_to(base)
    except ValueError:
        relative = file_path.name
    relative_text = _normalize_text(str(relative))
    name = _normalize_text(file_path.stem)
    search_text = f"{relative_text} {name}"

    provider = _normalize_text(_first_relative_part(relative) or "")
    role = _material_role(search_text)
    round_numbers = _round_numbers(search_text)
    round_number = round_numbers[0] if round_numbers else None
    form = _exam_form(search_text)
    domain = "교직논술" if "논술" in search_text else "교육과정"
    subject = _subject(search_text)

    return {
        "collection": "mock_exam",
        "document_type": "mock_exam_solution" if role in {"answer", "explanation"} else "mock_exam_question",
        "material_role": role,
        "exam_provider": provider or None,
        "exam_round": round_number,
        "exam_rounds": ",".join(str(number) for number in round_numbers),
        "exam_form": form,
        "exam_domain": domain,
        "subject": subject,
        "source_folder": str(base),
        "relative_path": relative_text,
        "study_priority": _study_priority(role, domain),
    }


def mock_exam_splitter_config() -> dict[str, Any]:
    return {
        "chunk_size": 450,
        "chunk_overlap": 90,
        "separators": [
            "\n\n",
            "\n[",
            "\n※",
            "\n<",
            "\n문제",
            "\n문항",
            "\n답안",
            "\n해설",
            "\n",
            "。 ",
            ". ",
            " ",
            "",
        ],
    }


def summarize_ingestion_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = sum(result.get("indexed") is True for result in results)
    skipped = sum(result.get("status") == "success" and result.get("indexed") is False for result in results)
    failed = sum(result.get("status") != "success" for result in results)
    chunks = sum(int(result.get("chunks_count") or 0) for result in results)
    pages = sum(int(result.get("pages_count") or 0) for result in results)
    return {
        "status": "success" if failed == 0 else "partial_success",
        "files_count": len(results),
        "indexed_count": indexed,
        "skipped_count": skipped,
        "failed_count": failed,
        "chunks_count": chunks,
        "pages_count": pages,
    }


def _first_relative_part(relative: Path | str) -> str | None:
    if isinstance(relative, Path) and len(relative.parts) > 1:
        return relative.parts[0]
    return None


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _material_role(text: str) -> str:
    if any(token in text for token in ["해설", "해설지", "참고자료"]):
        return "explanation"
    if any(token in text for token in ["답지", "답안", "모범답안", "정답"]):
        return "answer"
    if any(token in text for token in ["문제", "A형", "B형", "a형", "b형"]):
        return "question"
    if any(token in text for token in ["교육과정", "논술", "모의고사"]) or re.search(r"\d{1,2}\s*회", text):
        return "question"
    return "reference"


def _round_numbers(text: str) -> list[int]:
    values: list[int] = []
    for first, second in re.findall(r"(\d{1,2})(?:\s*[-,]\s*(\d{1,2}))?\s*회", text):
        for value in [first, second]:
            if not value:
                continue
            number = int(value)
            if number not in values:
                values.append(number)
    return values


def _exam_form(text: str) -> str | None:
    if re.search(r"(?:^|[\s_\-(])A(?:형|[\s_\-.)]|$)", text, re.IGNORECASE):
        return "A"
    if re.search(r"(?:^|[\s_\-(])B(?:형|[\s_\-.)]|$)", text, re.IGNORECASE):
        return "B"
    if "교육과정ab" in text.lower():
        return "A/B"
    return None


def _subject(text: str) -> str | None:
    for keyword, subject in SUBJECT_KEYWORDS.items():
        if keyword in text:
            return subject
    return None


def _study_priority(role: str, domain: str) -> int:
    if role == "explanation":
        return 1
    if role == "answer":
        return 2
    if domain == "교직논술":
        return 3
    return 4

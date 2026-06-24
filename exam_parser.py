#!/usr/bin/env python3
"""2015~2026 초등학교 교육과정 A/B 시험지 파서.

텍스트 PDF는 pypdf로 처리하고, 텍스트 레이어가 빈약한 스캔 PDF는
노트북에서 사용한 OpenCV 전처리 + EasyOCR 방식으로 자동 전환한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pypdf import PdfReader
from ocr_service import OCRService


EXAM_FILE_RE = re.compile(
    r"^(?P<year>20(?:1[5-9]|2[0-6]))학년도_"
    r"(?:초등학교_)?(?:교육|교직)과정(?P<form>[AB])\.pdf$"
)
HEADER_RE = re.compile(r"^초등학교\s*교육과정\s*[AB]\s*\(\d+면\s*중\s*\d+\s*면\)$")
QUESTION_RE = re.compile(r"^(?P<number>\d{1,2})\.\s*(?P<body>.*)$")
SUBQUESTION_RE = re.compile(r"^(?P<number>\d{1,2})\)\s*(?P<body>.*)$")
SCORE_RE = re.compile(r"\[(?P<score>\d+)점\]")
CURRICULUM_RE = re.compile(r"(?P<version>2007|2009|2015|2022)\s*개정(?:\s*교육과정)?")
ACHIEVEMENT_CODE_RE = re.compile(r"\[(?:\d{1,2})?[가-힣]{1,5}\d{2}-\d{2}\]")
VISUAL_REF_RE = re.compile(
    r"(?:\[(?:그림|자료|표|악보)\s*\d*\]|<(?:그림|자료|표|제재곡)[^>]*>)"
)

SUBJECT_ALIASES = {
    "국어": "국어",
    "수학": "수학",
    "사회": "사회",
    "과학": "과학",
    "영어": "영어",
    "음악": "음악",
    "미술": "미술",
    "체육": "체육",
    "실과": "실과",
    "도덕": "도덕",
    "총론창의적체험활동": "총론·창의적 체험활동",
    "총론･창의적체험활동": "총론·창의적 체험활동",
    "바른생활": "바른 생활",
    "슬기로운생활": "슬기로운 생활",
    "즐거운생활": "즐거운 생활",
    "통합교과": "통합교과",
}


@dataclass
class PageText:
    page: int
    text: str
    method: str
    char_count: int


@dataclass
class SubQuestion:
    number: int
    score: int | None
    text: str


@dataclass
class Question:
    id: str
    number: int
    subject: str | None
    score: int | None
    pages: list[int]
    curriculum_mentions: list[str]
    achievement_codes: list[str]
    visual_references: list[str]
    needs_visual_context: bool
    text: str
    subquestions: list[SubQuestion]


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def compact(value: str) -> str:
    return re.sub(r"[\s·ㆍ･・]", "", value).strip()


def normalize_text(value: str) -> str:
    value = nfc(value).replace("\u00a0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def detect_subject(line: str) -> str | None:
    key = compact(line)
    # 과목 머리말만 인정하여 본문 속 '수학' 같은 단어의 오탐을 막는다.
    if len(key) > 14:
        return None
    return SUBJECT_ALIASES.get(key)


def native_pages(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(pdf_path)
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = normalize_text(page.extract_text() or "")
        pages.append(PageText(number, text, "pypdf", len(text)))
    return pages


def needs_ocr(pages: Sequence[PageText], minimum_chars: int = 450) -> bool:
    if not pages:
        return True
    useful = [p.char_count for p in pages]
    return sum(useful) / len(useful) < minimum_chars or sum(c >= minimum_chars for c in useful) < len(useful) / 2


def _easyocr_lines(results: Sequence[Sequence[object]]) -> str:
    """EasyOCR bbox 결과를 위→아래, 왼쪽→오른쪽 문장으로 재조립한다."""
    items = []
    for bbox, text, confidence in results:
        if float(confidence) < 0.35 or not str(text).strip():
            continue
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        items.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), str(text).strip()))
    items.sort(key=lambda item: (item[0], item[1]))

    rows: list[dict[str, object]] = []
    for y, x, height, text in items:
        tolerance = max(10.0, height * 0.65)
        row = next((r for r in reversed(rows[-4:]) if abs(float(r["y"]) - y) <= tolerance), None)
        if row is None:
            rows.append({"y": y, "parts": [(x, text)]})
        else:
            row["parts"].append((x, text))
    return "\n".join(" ".join(text for _, text in sorted(row["parts"])) for row in rows)


def ocr_pages(pdf_path: Path, dpi: int = 300) -> list[PageText]:
    """PyMuPDF 렌더링 후 노트북의 전처리 절차로 EasyOCR을 실행한다."""
    try:
        import cv2
        import easyocr
        import fitz
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OCR에는 pymupdf, easyocr, opencv-python-headless, numpy가 필요합니다. "
            "`pip install -r requirements-parser-ocr.txt`를 실행하세요."
        ) from exc

    reader = easyocr.Reader(["ko", "en"], gpu=False)
    document = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    for number, page in enumerate(document, 1):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text = normalize_text(_easyocr_lines(reader.readtext(binary, detail=1, paragraph=False)))
        pages.append(PageText(number, text, "easyocr", len(text)))
    document.close()
    return pages


def extract_pages(pdf_path: Path, ocr: str = "auto", dpi: int = 250) -> tuple[list[PageText], list[str]]:
    pages = native_pages(pdf_path)
    warnings: list[str] = []
    use_ocr = ocr == "always" or (ocr == "auto" and needs_ocr(pages))
    if use_ocr:
        try:
            backend = "auto" if ocr == "auto" else "tesseract"
            extracted = OCRService(
                cache_dir="ocr_cache", backend=backend, dpi=dpi
            ).extract_pdf(pdf_path, force=(ocr == "always"))
            pages = [
                PageText(page.page, page.text, page.method, len(page.text))
                for page in extracted.pages
            ]
            warnings.extend(extracted.warnings)
        except RuntimeError as exc:
            if ocr == "always":
                raise
            warnings.append(str(exc))
            warnings.append("텍스트 품질이 낮지만 OCR을 실행하지 못해 pypdf 결과를 사용했습니다.")
    return pages, warnings


def _lines_with_context(pages: Sequence[PageText]) -> list[tuple[int, str, str | None]]:
    output = []
    subject = None
    for page in pages:
        page_lines = page.text.splitlines()
        for index, line in enumerate(page_lines):
            if HEADER_RE.match(line) or line in {"<수고하셨습니다.>", "(이하 여백)"}:
                continue
            found = detect_subject(line)
            # 과목 머리말은 다음 행에 새 대문항이 시작될 때만 인정한다.
            # 표/제시문 안에 홀로 놓인 '음악', '수학' 등의 오탐을 방지한다.
            next_line = page_lines[index + 1] if index + 1 < len(page_lines) else ""
            if found and QUESTION_RE.match(next_line):
                subject = found
                continue
            output.append((page.page, line, subject))
    return output


def _candidate_has_score(lines: Sequence[tuple[int, str, str | None]], index: int) -> bool:
    snippet = " ".join(line for _, line, _ in lines[index : index + 12])
    return bool(SCORE_RE.search(snippet))


def question_boundaries(lines: Sequence[tuple[int, str, str | None]], max_questions: int = 20) -> list[int]:
    """예상 번호를 순서대로 찾고 배점이 가까운 후보를 우선한다."""
    starts: list[int] = []
    cursor = 0
    for expected in range(1, max_questions + 1):
        candidates = []
        for index in range(cursor, len(lines)):
            match = QUESTION_RE.match(lines[index][1])
            if match and int(match.group("number")) == expected:
                candidates.append(index)
        if not candidates:
            break
        start = next((i for i in candidates if _candidate_has_score(lines, i)), candidates[0])
        starts.append(start)
        cursor = start + 1
    return starts


def parse_subquestions(text_lines: Sequence[str]) -> list[SubQuestion]:
    starts = []
    for index, line in enumerate(text_lines):
        match = SUBQUESTION_RE.match(line)
        if not match:
            continue
        nearby = " ".join(text_lines[index : index + 6])
        if SCORE_RE.search(nearby):
            starts.append(index)
    output = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(text_lines)
        block = "\n".join(text_lines[start:end]).strip()
        match = SUBQUESTION_RE.match(text_lines[start])
        score = SCORE_RE.search(block)
        output.append(SubQuestion(int(match.group("number")), int(score.group("score")) if score else None, block))
    return output


def parse_questions(pages: Sequence[PageText], exam_id: str) -> list[Question]:
    lines = _lines_with_context(pages)
    starts = question_boundaries(lines)
    questions = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        text_lines = [line for _, line, _ in block]
        text = "\n".join(text_lines).strip()
        match = QUESTION_RE.match(text_lines[0])
        score = SCORE_RE.search(" ".join(text_lines[:12]))
        questions.append(
            Question(
                id=f"{exam_id}-Q{int(match.group('number')):02d}",
                number=int(match.group("number")),
                subject=block[0][2],
                score=int(score.group("score")) if score else None,
                pages=sorted({page for page, _, _ in block}),
                curriculum_mentions=sorted(
                    {m.group("version") for m in CURRICULUM_RE.finditer(text)}
                ),
                achievement_codes=sorted(set(ACHIEVEMENT_CODE_RE.findall(text))),
                visual_references=list(dict.fromkeys(VISUAL_REF_RE.findall(text))),
                needs_visual_context=bool(VISUAL_REF_RE.search(text)),
                text=text,
                subquestions=parse_subquestions(text_lines),
            )
        )
    return questions


def parse_exam(pdf_path: Path, ocr: str = "auto", dpi: int = 300) -> dict[str, object]:
    filename = nfc(pdf_path.name)
    match = EXAM_FILE_RE.match(filename)
    if not match:
        raise ValueError(f"시험지 파일명 형식을 인식할 수 없습니다: {filename}")
    pages, warnings = extract_pages(pdf_path, ocr=ocr, dpi=dpi)
    exam_id = f"{match.group('year')}-{match.group('form')}"
    questions = parse_questions(pages, exam_id=exam_id)
    # 2026학년도부터 시험 편제가 10개 대문항으로 변경되었다.
    expected_count = 10 if int(match.group("year")) >= 2026 else 11
    expected = set(range(1, expected_count + 1))
    found = {q.number for q in questions}
    if found != expected:
        warnings.append(
            f"대문항 번호 점검 필요: 발견={sorted(found)}, 예상=1~{expected_count}"
        )
    return {
        "id": exam_id,
        "year": int(match.group("year")),
        "form": match.group("form"),
        "source": str(pdf_path),
        "page_count": len(pages),
        "extraction_methods": sorted({page.method for page in pages}),
        "warnings": warnings,
        "questions": [asdict(question) for question in questions],
        "pages": [asdict(page) for page in pages],
    }


def discover_pdfs(root: Path, years: set[int], forms: set[str]) -> list[Path]:
    found = []
    for path in root.glob("*.pdf"):
        match = EXAM_FILE_RE.match(nfc(path.name))
        if match and int(match.group("year")) in years and match.group("form") in forms:
            found.append(path)
    return sorted(found, key=lambda p: (EXAM_FILE_RE.match(nfc(p.name)).group("year"), nfc(p.name)))


def write_outputs(exams: Sequence[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for exam in exams:
        stem = f"{exam['year']}_교육과정{exam['form']}"
        (output_dir / f"{stem}.json").write_text(
            json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    with (output_dir / "questions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "id", "year", "form", "question", "subject", "score", "pages",
            "curriculum_mentions", "achievement_codes", "visual_references",
            "needs_visual_context", "text",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for exam in exams:
            for question in exam["questions"]:
                writer.writerow(
                    {
                        "id": question["id"],
                        "year": exam["year"],
                        "form": exam["form"],
                        "question": question["number"],
                        "subject": question["subject"] or "",
                        "score": question["score"] if question["score"] is not None else "",
                        "pages": ",".join(map(str, question["pages"])),
                        "curriculum_mentions": ",".join(question["curriculum_mentions"]),
                        "achievement_codes": ",".join(question["achievement_codes"]),
                        "visual_references": ",".join(question["visual_references"]),
                        "needs_visual_context": question["needs_visual_context"],
                        "text": question["text"],
                    }
                )

    summary = [
        {
            "year": exam["year"],
            "form": exam["form"],
            "pages": exam["page_count"],
            "questions": len(exam["questions"]),
            "methods": exam["extraction_methods"],
            "warnings": exam["warnings"],
        }
        for exam in exams
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_years(value: str) -> set[int]:
    years: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    invalid = years - set(range(2015, 2027))
    if invalid:
        raise argparse.ArgumentTypeError(f"지원 범위(2015~2026) 밖의 연도: {sorted(invalid)}")
    return years


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("."), help="시험지 PDF 폴더")
    parser.add_argument("-o", "--output", type=Path, default=Path("parsed_exams"), help="결과 폴더")
    parser.add_argument("--years", type=parse_years, default=set(range(2015, 2027)))
    parser.add_argument("--forms", choices=["A", "B", "AB"], default="AB")
    parser.add_argument("--ocr", choices=["auto", "never", "always"], default="auto")
    parser.add_argument("--dpi", type=int, default=250, help="OCR 렌더링 해상도")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forms = set(args.forms)
    pdfs = discover_pdfs(args.input, args.years, forms)
    if not pdfs:
        print("조건에 맞는 시험지 PDF를 찾지 못했습니다.", file=sys.stderr)
        return 1
    exams = []
    for index, pdf in enumerate(pdfs, 1):
        print(f"[{index}/{len(pdfs)}] {nfc(pdf.name)}", file=sys.stderr)
        try:
            exams.append(parse_exam(pdf, ocr=args.ocr, dpi=args.dpi))
        except Exception as exc:
            print(f"  실패: {exc}", file=sys.stderr)
    write_outputs(exams, args.output)
    print(f"{len(exams)}개 시험지 결과 저장: {args.output}", file=sys.stderr)
    return 0 if len(exams) == len(pdfs) else 2


if __name__ == "__main__":
    raise SystemExit(main())

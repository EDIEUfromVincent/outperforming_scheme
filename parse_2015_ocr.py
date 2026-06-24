#!/usr/bin/env python3
"""2015학년도 교육과정 A/B 시험지만 OCR로 보강 파싱한다."""

from __future__ import annotations

import json
import re
from pathlib import Path

from exam_parser import (
    ACHIEVEMENT_CODE_RE,
    CURRICULUM_RE,
    SCORE_RE,
    VISUAL_REF_RE,
    discover_pdfs,
    parse_exam,
)


TOP_LEVEL_RE = re.compile(r"^\s*(?P<number>\d{1,2})\.\s*(?P<body>.*)")


def subject_for(form: str, number: int) -> str | None:
    if form == "A":
        if number <= 3:
            return "국어"
        if number <= 6:
            return "수학"
        if number <= 8:
            return "사회"
        if number <= 10:
            return "과학"
        return "통합교과"
    if number == 1:
        return "총론·창의적 체험활동"
    if number == 2:
        return "바른 생활"
    if number == 3:
        return "도덕"
    if number == 4:
        return "실과"
    if number in {5, 6}:
        return "체육"
    if number == 7:
        return "미술"
    if number == 8:
        return "음악"
    if number == 9:
        return "즐거운 생활"
    if number in {10, 11}:
        return "영어"
    return None


def fuzzy_questions(exam: dict) -> list[dict]:
    """2015 스캔 OCR 전용: 살아남은 대문항 번호와 페이지를 이용해 chunk를 복구한다."""
    numbered_lines = []
    for page in exam["pages"]:
        for line in page["text"].splitlines():
            numbered_lines.append((page["page"], line))

    starts = []
    last_number = 0
    for index, (_, line) in enumerate(numbered_lines):
        match = TOP_LEVEL_RE.match(line)
        if not match:
            continue
        number = int(match.group("number"))
        if not (1 <= number <= 11):
            continue
        # 하위 예시 번호가 뒤늦게 나오는 경우를 피하기 위해 대문항 번호는 증가할 때만 인정한다.
        if number <= last_number:
            continue
        starts.append((index, number))
        last_number = number

    if not starts:
        starts = [(0, 1)]
    elif starts[0][1] != 1:
        starts.insert(0, (0, 1))

    starts = add_2015_missing_boundaries(exam, starts, numbered_lines)

    questions = []
    for position, (start, number) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(numbered_lines)
        block = numbered_lines[start:end]
        text = "\n".join(line for _, line in block).strip()
        if len(text) < 80:
            continue
        score = SCORE_RE.search(text)
        pages = sorted({page for page, _ in block})
        questions.append({
            "id": f"{exam['id']}-Q{number:02d}",
            "number": number,
            "subject": subject_for(exam["form"], number),
            "score": int(score.group("score")) if score else None,
            "pages": pages,
            "curriculum_mentions": sorted({m.group("version") for m in CURRICULUM_RE.finditer(text)}),
            "achievement_codes": sorted(set(ACHIEVEMENT_CODE_RE.findall(text))),
            "visual_references": list(dict.fromkeys(VISUAL_REF_RE.findall(text))),
            "needs_visual_context": bool(VISUAL_REF_RE.search(text)),
            "text": text,
            "subquestions": [],
            "parser_note": "2015 OCR fuzzy boundary recovery",
        })
    return questions


def add_2015_missing_boundaries(exam: dict, starts: list[tuple[int, int]], lines: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """OCR이 대문항 번호를 떨어뜨린 2015 일부 지점을 시험 편제 메타데이터로 보정한다."""
    existing = {number for _, number in starts}
    additions: list[tuple[int, int]] = []
    form = exam["form"]

    def find_between(pattern: str, after_number: int, before_number: int) -> int | None:
        start_index = next((idx for idx, num in starts if num == after_number), None)
        end_index = next((idx for idx, num in starts if num == before_number), len(lines))
        if start_index is None:
            return None
        regex = re.compile(pattern)
        for idx in range(start_index + 1, end_index):
            if regex.search(lines[idx][1]):
                return idx
        return None

    if form == "A":
        if 6 not in existing:
            idx = find_between(r"다\s*음\s*은.*문\s*제.*해\s*결|규\s*칙", 5, 7)
            if idx is not None:
                additions.append((idx, 6))
        if 8 not in existing:
            idx = find_between(r"예\s*비\s*교\s*사|위\s*치|경\s*선", 7, 9)
            if idx is not None:
                additions.append((idx, 8))
    else:
        if 5 not in existing:
            idx = find_between(r"축\s*구|패\s*스|멘토\s*교사", 4, 6)
            if idx is not None:
                additions.append((idx, 5))
        if 7 not in existing:
            idx = find_between(r"미\s*술|색\s*의|색\s*천|작\s*품", 6, 8)
            if idx is not None:
                additions.append((idx, 7))

    return sorted([*starts, *additions], key=lambda item: item[0])


def main() -> int:
    output_dir = Path("parsed_exams_native")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = discover_pdfs(Path("."), {2015}, {"A", "B"})
    if not pdfs:
        raise SystemExit("2015학년도 교육과정 A/B PDF를 찾지 못했습니다.")

    results = []
    for pdf in pdfs:
        exam = parse_exam(pdf, ocr="always", dpi=250)
        if len(exam["questions"]) < 8:
            recovered = fuzzy_questions(exam)
            exam["warnings"].append(
                f"2015 OCR 보정 파서 적용: 기존 {len(exam['questions'])}문항 → {len(recovered)}문항"
            )
            exam["questions"] = recovered
        target = output_dir / f"{exam['year']}_교육과정{exam['form']}.json"
        target.write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({
            "source": str(pdf),
            "target": str(target),
            "form": exam["form"],
            "questions": len(exam["questions"]),
            "methods": exam["extraction_methods"],
            "warnings": exam["warnings"],
        })

    Path("parsed_exams_native/2015_ocr_report.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(row["questions"] > 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

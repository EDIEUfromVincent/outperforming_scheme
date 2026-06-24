#!/usr/bin/env python3
"""자동 대응표 중 검수 필요 항목을 초등임용 관점으로 우선순위화한다."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from comparison_service import CurriculumComparisonService


SUBJECT_WEIGHT = {
    "국어": 5,
    "수학": 5,
    "과학": 5,
    "사회": 4,
    "통합교과": 4,
    "도덕": 3,
    "영어": 3,
    "체육": 2,
    "음악": 2,
    "미술": 2,
    "실과": 2,
}
GRADE_WEIGHT = {"5-6": 3, "3-4": 2, "1-2": 2}
CHANGE_WEIGHT = {
    "semantic_match": 5,
    "new_or_unmatched": 4,
    "removed_or_unmatched": 4,
    "same_code_modified": 1,
    "unchanged": 0,
}
EXAM_LIKE_TERMS = [
    "토의", "토론", "읽기", "쓰기", "분수", "나눗셈", "소리", "전기", "회로",
    "민주주의", "인권", "평가", "피드백", "과정", "탐구", "실천", "생활",
    "감상", "표현", "건강", "경쟁", "소프트웨어", "절차", "지역",
]


def row_text(row: dict) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("code_2015", "text_2015", "code_2022", "text_2022")
    )


def priority(row: dict, related_exam_count: int) -> tuple[int, list[str]]:
    reasons = []
    score = 0
    subject_score = SUBJECT_WEIGHT.get(row["subject"], 1)
    grade_score = GRADE_WEIGHT.get(row["grade_band"], 1)
    change_score = CHANGE_WEIGHT.get(row["change_type"], 0)
    score += subject_score + grade_score + change_score
    reasons.append(f"교과가중치 {subject_score}")
    reasons.append(f"학년군가중치 {grade_score}")
    reasons.append(f"변화유형가중치 {change_score}")

    text = row_text(row)
    hit_terms = [term for term in EXAM_LIKE_TERMS if term in text]
    if hit_terms:
        boost = min(5, len(hit_terms))
        score += boost
        reasons.append(f"임용형 키워드 {', '.join(hit_terms[:5])}")
    if related_exam_count:
        boost = min(4, related_exam_count)
        score += boost
        reasons.append(f"관련 기출 {related_exam_count}개")
    if row["change_type"] in {"semantic_match", "new_or_unmatched", "removed_or_unmatched"}:
        reasons.append("자동 추정 대응")
    return score, reasons


def main() -> int:
    service = CurriculumComparisonService()
    review_rows = [row for row in service.mappings if row.get("review_required")]
    enriched = []
    for row in review_rows:
        query = row_text(row)[:1000]
        related_exam_count = len(service._related_exams(query, limit=6))
        score, reasons = priority(row, related_exam_count)
        if score >= 17:
            tier = "high"
        elif score >= 13:
            tier = "medium"
        else:
            tier = "low"
        enriched.append({
            "review_priority": tier,
            "priority_score": score,
            "review_reason": "; ".join(reasons),
            "related_exam_count": related_exam_count,
            **row,
        })

    enriched.sort(
        key=lambda row: (
            {"high": 0, "medium": 1, "low": 2}[row["review_priority"]],
            -row["priority_score"],
            row["subject"],
            row["grade_band"],
        )
    )
    summary = {
        "total_review_required": len(enriched),
        "by_priority": Counter(row["review_priority"] for row in enriched),
        "by_subject": Counter(row["subject"] for row in enriched),
        "by_change_type": Counter(row["change_type"] for row in enriched),
        "top_30": [
            {
                "review_priority": row["review_priority"],
                "priority_score": row["priority_score"],
                "subject": row["subject"],
                "grade_band": row["grade_band"],
                "change_type": row["change_type"],
                "code_2015": row.get("code_2015"),
                "code_2022": row.get("code_2022"),
                "related_exam_count": row["related_exam_count"],
                "review_reason": row["review_reason"],
            }
            for row in enriched[:30]
        ],
    }
    output_dir = Path("curriculum_mapping")
    (output_dir / "review_priorities.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "review_priority_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if enriched:
        with (output_dir / "review_priorities.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(enriched[0]))
            writer.writeheader()
            writer.writerows(enriched)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""초등임용 수험생 관점의 교과별 대표 질문 검증 하네스."""

from __future__ import annotations

import json
from pathlib import Path

from comparison_service import CurriculumComparisonService


CASES = [
    {"subject": "국어", "grade_band": "1-2", "query": "글자와 낱말 읽기"},
    {"subject": "국어", "grade_band": "3-4", "query": "중심 생각 찾기"},
    {"subject": "국어", "grade_band": "5-6", "query": "토의 토론"},
    {"subject": "수학", "grade_band": "1-2", "query": "두 자리 수 덧셈 뺄셈"},
    {"subject": "수학", "grade_band": "3-4", "query": "분수의 의미"},
    {"subject": "수학", "grade_band": "5-6", "query": "분수의 나눗셈"},
    {"subject": "과학", "grade_band": "3-4", "query": "소리의 성질"},
    {"subject": "과학", "grade_band": "5-6", "query": "전기 회로"},
    {"subject": "과학", "grade_band": "5-6", "query": "생물과 환경"},
    {"subject": "사회", "grade_band": "3-4", "query": "지역의 위치와 특성"},
    {"subject": "사회", "grade_band": "5-6", "query": "인권과 민주주의"},
    {"subject": "사회", "grade_band": "5-6", "query": "경제생활과 선택"},
    {"subject": "도덕", "grade_band": "3-4", "query": "도덕적 행동 실천"},
    {"subject": "도덕", "grade_band": "5-6", "query": "공동체와 책임"},
    {"subject": "실과", "grade_band": "5-6", "query": "소프트웨어와 절차적 사고"},
    {"subject": "실과", "grade_band": "5-6", "query": "생활 자원 관리"},
    {"subject": "체육", "grade_band": "3-4", "query": "건강 생활 습관"},
    {"subject": "체육", "grade_band": "5-6", "query": "경쟁 활동 전략"},
    {"subject": "음악", "grade_band": "3-4", "query": "노래 부르기와 표현"},
    {"subject": "음악", "grade_band": "5-6", "query": "음악 감상과 생활화"},
    {"subject": "미술", "grade_band": "3-4", "query": "관찰 표현"},
    {"subject": "미술", "grade_band": "5-6", "query": "작품 감상과 비평"},
    {"subject": "영어", "grade_band": "3-4", "query": "듣고 말하기"},
    {"subject": "영어", "grade_band": "5-6", "query": "읽기 쓰기"},
    {"subject": "통합교과", "grade_band": "1-2", "query": "학교와 생활"},
    {"subject": "통합교과", "grade_band": "1-2", "query": "계절과 생활"},
    {"subject": None, "grade_band": None, "query": "과정 중심 평가"},
    {"subject": None, "grade_band": None, "query": "학생 참여형 수업"},
    {"subject": None, "grade_band": None, "query": "성취기준 변화 수업 설계"},
    {"subject": None, "grade_band": None, "query": "평가 증거와 피드백"},
]


def _snippet(row: dict) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("code_2015", "text_2015", "code_2022", "text_2022")
    )[:500]


def main() -> int:
    service = CurriculumComparisonService()
    results = []
    for index, case in enumerate(CASES, 1):
        result = service.compare(
            subject=case["subject"],
            grade_band=case["grade_band"],
            query=case["query"],
            limit=5,
        )
        rows = result["comparisons"]
        scoped = all(
            (case["subject"] is None or row["subject"] == case["subject"])
            and (case["grade_band"] is None or row["grade_band"] == case["grade_band"])
            for row in rows
        )
        both_versions = any(row.get("text_2015") and row.get("text_2022") for row in rows)
        query_terms = [term for term in case["query"].replace("·", " ").split() if len(term) >= 2]
        hit_terms = [
            term for term in query_terms
            if any(term in _snippet(row) for row in rows)
        ]
        top_score = rows[0].get("query_score") if rows else 0.0
        has_change_signal = both_versions or any(row.get("review_required") for row in rows)
        passed = bool(rows) and scoped and has_change_signal and (
            bool(hit_terms)
            or case["subject"] is None
            or float(top_score or 0.0) >= 0.08
        )
        results.append({
            "case_id": index,
            **case,
            "passed": passed,
            "comparison_count": len(rows),
            "related_exam_count": len(result["related_exams"]),
            "hit_terms": hit_terms,
            "both_versions_found": both_versions,
            "needs_human_review": any(row.get("review_required") for row in rows),
            "top_rows": [
                {
                    "subject": row["subject"],
                    "grade_band": row["grade_band"],
                    "change_type": row["change_type"],
                    "code_2015": row.get("code_2015"),
                    "code_2022": row.get("code_2022"),
                    "query_score": row.get("query_score"),
                    "review_required": row.get("review_required"),
                }
                for row in rows[:3]
            ],
        })

    output = {
        "passed": all(row["passed"] for row in results),
        "passed_count": sum(row["passed"] for row in results),
        "total_count": len(results),
        "results": results,
    }
    Path("representative_question_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""프로젝트 1의 OCR·대응표·검색 품질을 대표 질문으로 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

from comparison_service import CurriculumComparisonService


CASES = [
    {"name": "국어 토의토론", "subject": "국어", "grade_band": "5-6", "query": "토의 토론"},
    {"name": "수학 분수 나눗셈", "subject": "수학", "grade_band": "5-6", "query": "분수의 나눗셈"},
    {"name": "과학 소리", "subject": "과학", "grade_band": "3-4", "query": "소리의 성질"},
    {"name": "통합교과 생활", "subject": "통합교과", "grade_band": "1-2", "query": "학교와 생활"},
    {"name": "과정중심평가", "subject": None, "grade_band": None, "query": "과정 중심 평가"},
]


def main() -> int:
    checks = []
    ocr_report_path = Path("ocr_cache/report.json")
    if ocr_report_path.exists():
        report = json.loads(ocr_report_path.read_text(encoding="utf-8"))
        failures = [row for row in report["results"] if row["status"] != "success"]
        checks.append({"check": "ocr_batch", "passed": not failures, "failures": failures})
    else:
        checks.append({"check": "ocr_batch", "passed": False, "reason": "report missing"})

    mapping_summary_path = Path("curriculum_mapping/summary.json")
    if mapping_summary_path.exists():
        summary = json.loads(mapping_summary_path.read_text(encoding="utf-8"))
        passed = summary["standards_2015"] > 100 and summary["standards_2022"] > 100
        checks.append({"check": "mapping_volume", "passed": passed, **summary})
    else:
        checks.append({"check": "mapping_volume", "passed": False, "reason": "summary missing"})

    service = CurriculumComparisonService()
    for case in CASES:
        result = service.compare(
            subject=case["subject"], grade_band=case["grade_band"],
            query=case["query"], limit=5,
        )
        both_versions = any(
            row.get("text_2015") and row.get("text_2022")
            for row in result["comparisons"]
        )
        checks.append({
            "check": case["name"],
            "passed": result["count"] > 0 and both_versions,
            "comparison_count": result["count"],
            "related_exam_count": len(result["related_exams"]),
        })

    output = {
        "passed": all(check["passed"] for check in checks),
        "passed_count": sum(check["passed"] for check in checks),
        "total_count": len(checks), "checks": checks,
    }
    Path("validation_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

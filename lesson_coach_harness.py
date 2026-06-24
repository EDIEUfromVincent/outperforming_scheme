#!/usr/bin/env python3
"""수업-임용 통합 코치 에이전트 하네스."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app


CASES = [
    {
        "name": "5학년 수학 분수의 나눗셈",
        "query": "내일 5학년 수학 분수의 나눗셈 수업을 해야 하는데, 임용 관점에서 성취기준 변화와 기출 포인트까지 같이 공부하고 싶어.",
        "subject": "수학",
        "grade": 5,
    },
    {
        "name": "3학년 과학 소리의 성질",
        "query": "3학년 과학 소리의 성질 수업을 설계하면서 평가 증거와 임용 기출 관점을 같이 정리해줘.",
        "subject": "과학",
        "grade": 3,
    },
    {
        "name": "6학년 국어 토의 토론",
        "query": "6학년 국어 토의 토론 수업에서 학생 반응을 예상하고 임용 인출 질문까지 만들고 싶어.",
        "subject": "국어",
        "grade": 6,
    },
]


def main() -> int:
    client = TestClient(app)
    results = []
    for case in CASES:
        response = client.post("/lesson-coach", json={**case, "limit": 5})
        data = response.json()
        checks = {
            "status_200": response.status_code == 200,
            "route_subject": data.get("route", {}).get("subject") == case["subject"],
            "route_grade": data.get("route", {}).get("grade") == case["grade"],
            "governance_docs": len(data.get("governance_docs", [])) > 0,
            "comparisons": len(data.get("subject_agent", {}).get("comparisons", [])) > 0,
            "grade_lens": bool(data.get("grade_student_agent", {}).get("focus")),
            "answer_sections": all(
                marker in data.get("answer", "")
                for marker in ["①", "②", "③", "④", "⑤", "⑥", "⑦"]
            ),
        }
        results.append({
            "name": case["name"],
            "passed": all(checks.values()),
            "checks": checks,
            "route": data.get("route"),
            "audit": data.get("audit"),
            "related_exam_count": len(data.get("subject_agent", {}).get("related_exams", [])),
        })

    output = {
        "passed": all(row["passed"] for row in results),
        "passed_count": sum(row["passed"] for row in results),
        "total_count": len(results),
        "results": results,
    }
    Path("lesson_coach_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

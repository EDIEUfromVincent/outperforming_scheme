#!/usr/bin/env python3
"""학습 기록·이동평균·복습 일정·변형 문제 하네스."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app


def main() -> int:
    client = TestClient(app)
    seed = client.post("/learning/seed-demo")
    attempt = client.post("/learning/attempt", json={
        "standard_code": "6수01-11",
        "subject": "수학",
        "grade_band": "5-6",
        "is_correct": False,
        "confidence": 2,
        "time_spent_sec": 120,
        "question_text": "분수의 나눗셈 원리를 설명하시오.",
        "user_answer": "절차만 설명함",
        "source_type": "harness",
    })
    metrics = client.get("/learning/metrics", params={"days": 60})
    aggregate = client.get("/learning/metrics/aggregate", params={"days": 180, "period": "W"})
    weak = client.get("/learning/weak-standards", params={"limit": 5})
    due = client.get("/learning/review-due", params={"limit": 5})
    due_rows = due.json().get("due_reviews", []) if due.status_code == 200 else []
    complete = client.post("/learning/review-complete", json={
        "standard_code": due_rows[0]["standard_code"] if due_rows else "6수01-11",
    })
    variant = client.post("/learning/generate-variant", json={
        "standard_code": "6수01-11",
        "subject": "수학",
        "grade_band": "5-6",
        "weakness_note": "분수의 나눗셈 원리 설명이 약함",
    })
    data = {
        "seed_status": seed.status_code,
        "attempt_status": attempt.status_code,
        "metrics_status": metrics.status_code,
        "aggregate_status": aggregate.status_code,
        "weak_status": weak.status_code,
        "due_status": due.status_code,
        "complete_status": complete.status_code,
        "variant_status": variant.status_code,
        "metrics_count": len(metrics.json().get("metrics", [])),
        "aggregate_count": len(aggregate.json().get("metrics", [])),
        "weak_count": len(weak.json().get("weak_standards", [])),
        "due_count": len(due.json().get("due_reviews", [])),
        "completed_reviews": complete.json().get("completed_reviews", 0),
        "variant_has_text": "변형 문제" in variant.json().get("quiz_text", ""),
    }
    checks = {
        "all_status_ok": all(data[key] == 200 for key in [
            "seed_status", "attempt_status", "metrics_status", "aggregate_status",
            "weak_status", "due_status", "complete_status", "variant_status"
        ]),
        "metrics_present": data["metrics_count"] > 0,
        "aggregate_present": data["aggregate_count"] > 0,
        "weak_present": data["weak_count"] > 0,
        "due_present": data["due_count"] > 0,
        "complete_present": data["completed_reviews"] >= 0,
        "variant_present": data["variant_has_text"],
    }
    output = {
        "passed": all(checks.values()),
        "checks": checks,
        "data": data,
    }
    Path("learning_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

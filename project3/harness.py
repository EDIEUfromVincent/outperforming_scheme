#!/usr/bin/env python3
"""Project 3 로컬 검증."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app


def main() -> int:
    client = TestClient(app)
    sources = client.get("/sources", params={"include_regions": False})
    status = client.get("/email/status")
    preview = client.post("/digest/preview", json={
        "recipient": "ohjinwoo9696@gmail.com",
        "include_regions": False,
        "send": False,
        "seed": "harness",
    })
    email_no_send = client.post("/digest/email", json={
        "recipient": "ohjinwoo9696@gmail.com",
        "include_regions": False,
        "send": False,
        "seed": "harness",
    })
    preview_json = preview.json() if preview.status_code == 200 else {}
    body = preview_json.get("body", "")
    checks = {
        "sources_ok": sources.status_code == 200 and len(sources.json().get("sources", [])) >= 4,
        "email_status_ok": status.status_code == 200,
        "preview_ok": preview.status_code == 200,
        "no_send_ok": email_no_send.status_code == 200 and email_no_send.json().get("sent") is False,
        "has_past_question": "오늘의 기출문제" in body,
        "has_practice_question": "오늘의 예비문제" in body,
        "has_news_section": "오늘의 임용 공지" in body,
        "recipient_ok": preview_json.get("recipient") == "ohjinwoo9696@gmail.com",
    }
    output = {
        "passed": all(checks.values()),
        "checks": checks,
        "data": {
            "sources_status": sources.status_code,
            "email_status": status.status_code,
            "preview_status": preview.status_code,
            "email_no_send_status": email_no_send.status_code,
            "subject": preview_json.get("subject"),
            "body_chars": len(body),
            "notice_count": preview_json.get("notices", {}).get("count"),
            "notice_warnings": len(preview_json.get("notices", {}).get("warnings", [])),
        },
    }
    Path("project3_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

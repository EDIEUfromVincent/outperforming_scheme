#!/usr/bin/env python3
"""Streamlit 화면 하네스: 필터 선택·질문 입력·비교 버튼 클릭을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main() -> int:
    checks = []
    app = AppTest.from_file("app.py", default_timeout=60).run()

    checks.append({
        "check": "title",
        "passed": bool(app.title) and "초등임용" in app.title[0].value,
        "value": app.title[0].value if app.title else None,
    })
    checks.append({
        "check": "comparison_filters",
        "passed": (
            len(app.selectbox) >= 3
            and "수학" in app.selectbox[0].options
            and "5-6" in app.selectbox[1].options
        ),
        "subjects": app.selectbox[0].options if len(app.selectbox) >= 1 else [],
        "grade_bands": app.selectbox[1].options if len(app.selectbox) >= 2 else [],
    })

    app.selectbox[0].select("수학").run()
    app.selectbox[1].select("5-6").run()
    app.text_input[0].input("분수의 나눗셈 수업 설계").run()
    app.button[0].click().run(timeout=120)

    markdown_text = "\n".join(item.value for item in app.markdown if getattr(item, "value", None))
    checks.append({
        "check": "comparison_click",
        "passed": (
            not app.error
            and "6수01-11" in markdown_text
            and "분수의 나눗셈" in markdown_text
            and "변화 유형" in markdown_text
        ),
        "errors": [item.value for item in app.error],
        "expander_count": len(app.expander),
        "markdown_count": len(app.markdown),
    })
    checks.append({
        "check": "clean_text",
        "passed": "\x00" not in markdown_text,
        "control_char_found": "\x00" in markdown_text,
    })

    output = {
        "passed": all(row["passed"] for row in checks),
        "passed_count": sum(row["passed"] for row in checks),
        "total_count": len(checks),
        "checks": checks,
    }
    Path("streamlit_ui_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

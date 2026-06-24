"""Harness for multi-document upload selection and comparison APIs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import main


REPORT_PATH = Path("multi_document_compare_report.json")


def main_harness() -> int:
    client = TestClient(main.app)
    checks: dict[str, bool] = {}

    health = client.get("/health")
    features = health.json().get("features", {}) if health.status_code == 200 else {}
    checks["health_ok"] = health.status_code == 200
    checks["multi_feature_flag"] = bool(features.get("multi_document_query"))
    checks["graph_feature_flag"] = bool(features.get("document_compare_graph"))

    openapi = client.get("/openapi.json")
    schemas = openapi.json().get("components", {}).get("schemas", {}) if openapi.status_code == 200 else {}
    query_props = schemas.get("QueryRequest", {}).get("properties", {})
    checks["query_accepts_document_ids"] = "document_ids" in query_props
    checks["compare_endpoint_present"] = "/documents/compare" in openapi.json().get("paths", {})

    uploaded = client.get("/uploaded-documents")
    uploaded_docs = [
        doc for doc in uploaded.json().get("documents", [])
        if doc.get("document_id")
    ] if uploaded.status_code == 200 else []
    selected_ids = [doc["document_id"] for doc in uploaded_docs[:2]]
    checks["has_two_uploaded_docs"] = len(selected_ids) >= 2

    compare_status = None
    compare_payload = {}
    if len(selected_ids) >= 2:
        compare = client.post(
            "/documents/compare",
            json={
                "document_ids": selected_ids,
                "question": "두 문서를 비교하고 핵심 공통점과 차이점을 요약해줘.",
                "k_per_doc": 3,
            },
        )
        compare_status = compare.status_code
        compare_payload = compare.json() if compare.status_code == 200 else {}
        checks["compare_status_ok"] = compare_status == 200
        checks["compare_has_answer"] = bool(compare_payload.get("answer"))
        checks["compare_balanced_summaries"] = len(compare_payload.get("document_summaries", [])) == 2
        checks["compare_uses_graph"] = compare_payload.get("workflow") == "langgraph"

        retrieved = client.post(
            "/documents",
            json={
                "question": "비교 요약",
                "document_ids": selected_ids,
            },
        )
        checks["multi_documents_status_ok"] = retrieved.status_code == 200
        checks["multi_documents_returned"] = len(retrieved.json().get("documents", [])) >= 2
    else:
        checks["compare_status_ok"] = False
        checks["compare_has_answer"] = False
        checks["compare_balanced_summaries"] = False
        checks["compare_uses_graph"] = False
        checks["multi_documents_status_ok"] = False
        checks["multi_documents_returned"] = False

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "data": {
            "uploaded_doc_count": len(uploaded_docs),
            "selected_document_ids": selected_ids,
            "compare_status": compare_status,
            "workflow": compare_payload.get("workflow"),
            "summary_count": len(compare_payload.get("document_summaries", [])),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main_harness())

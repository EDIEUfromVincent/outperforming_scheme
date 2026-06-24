"""Harness for the LangGraph PDF upload ingestion workflow."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from document_ingestion_graph import run_pdf_ingestion_graph
import main
from ocr_service import OCRDocument, OCRPage


REPORT_PATH = Path("document_ingestion_report.json")
SAMPLE_PDF = Path("uploaded_files/임고비전 교직논술(해설).pdf")


def main_harness() -> int:
    client = TestClient(main.app)
    checks: dict[str, bool] = {}

    health = client.get("/health")
    features = health.json().get("features", {}) if health.status_code == 200 else {}
    checks["health_ok"] = health.status_code == 200
    checks["ingestion_graph_flag"] = bool(features.get("document_ingestion_graph"))
    checks["langgraph_persistence_flag"] = bool(features.get("langgraph_persistence"))

    if SAMPLE_PDF.exists():
        with SAMPLE_PDF.open("rb") as file:
            upload = client.post(
                "/upload-document",
                files={
                    "file": (
                        "graph_ingestion_duplicate_check.pdf",
                        file,
                        "application/pdf",
                    )
                },
            )
    else:
        upload = None

    payload = upload.json() if upload is not None and upload.status_code == 200 else {}
    trace = payload.get("trace", [])
    checks["sample_pdf_exists"] = SAMPLE_PDF.exists()
    checks["upload_status_ok"] = upload is not None and upload.status_code == 200
    checks["workflow_is_graph"] = payload.get("workflow") == "langgraph_pdf_ingestion"
    checks["thread_id_returned"] = bool(payload.get("thread_id"))
    checks["quality_report_returned"] = bool(payload.get("quality_report"))
    checks["trace_has_expected_nodes"] = [
        item.get("node") for item in trace
    ] == ["extract_ocr", "classify", "quality_check", "split", "index"]
    checks["duplicate_skipped"] = payload.get("indexed") is False
    new_document_result = _fake_new_document_run()
    new_document_trace = new_document_result.get("trace", [])
    checks["new_document_indexed"] = new_document_result.get("indexed") is True
    checks["new_document_chunks"] = new_document_result.get("chunks_count", 0) >= 2
    checks["new_document_trace_indexes"] = (
        new_document_trace[-1].get("node") == "index"
        and new_document_trace[-1].get("status") == "ok"
    )

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "data": {
            "upload_status": upload.status_code if upload is not None else None,
            "workflow": payload.get("workflow"),
            "thread_id": payload.get("thread_id"),
            "indexed": payload.get("indexed"),
            "trace": trace,
            "quality_report": payload.get("quality_report"),
            "new_document": {
                "indexed": new_document_result.get("indexed"),
                "chunks_count": new_document_result.get("chunks_count"),
                "trace": new_document_trace,
            },
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def _fake_new_document_run() -> dict:
    class FakeOCRService:
        def extract_pdf(self, pdf_path: str) -> OCRDocument:
            text = (
                "초등임용 모의고사 해설 자료입니다. "
                "성취기준, 오개념, 발문, 평가 장면을 비교할 수 있게 구성합니다. "
                "문서 업로드 그래프는 OCR, 품질검사, 분류, 청크 분할, 색인으로 이어집니다. "
            ) * 12
            return OCRDocument(
                document_id="fake-new-document",
                source=pdf_path,
                pages=[
                    OCRPage(
                        page=1,
                        text=text,
                        method="pypdf",
                        confidence=None,
                        native_char_count=len(text),
                    )
                ],
                warnings=[],
            )

    class FakeService:
        def __init__(self) -> None:
            self.ocr_service = FakeOCRService()
            self.indexed_document_ids = set()
            self.indexed_chunks = []

        def _document_metadata(self, pdf_path: str, document_id: str) -> dict:
            return {
                "source": pdf_path,
                "document_id": document_id,
                "document_type": "mock_exam",
                "collection": "harness",
            }

        def _index_chunks(self, chunks, document_id: str) -> None:
            self.indexed_chunks = chunks
            self.indexed_document_ids.add(document_id)

    service = FakeService()
    return run_pdf_ingestion_graph(
        service,
        pdf_path="fake-new-document.pdf",
        chunk_size=240,
        chunk_overlap=40,
        thread_id="document-ingest-harness-new",
    )


if __name__ == "__main__":
    raise SystemExit(main_harness())

"""CLI for ingesting the 임용-모의고사 folder into the local FAISS index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from exam_material_ingestion import (
    DEFAULT_MOCK_EXAM_DIR,
    infer_mock_exam_metadata,
    list_mock_exam_pdfs,
    mock_exam_splitter_config,
    summarize_ingestion_results,
)
from langchain_service import LangChainService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_MOCK_EXAM_DIR)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("mock_exam_ingest_report.json"))
    args = parser.parse_args()

    load_dotenv()
    root = args.root.expanduser()
    pdfs = list_mock_exam_pdfs(root)
    offset = max(0, args.offset)
    pdfs = pdfs[offset:]
    if args.limit is not None:
        pdfs = pdfs[: max(0, args.limit)]

    splitter_config = mock_exam_splitter_config()
    if args.dry_run:
        report = {
            "status": "dry_run",
            "root": str(root),
            "offset": offset,
            "files_count": len(pdfs),
            "splitter": splitter_config,
            "preview": [
                {
                    "filename": pdf.name,
                    "relative_path": str(pdf.relative_to(root)),
                    "metadata": infer_mock_exam_metadata(pdf, root),
                }
                for pdf in pdfs[:50]
            ],
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    service = LangChainService()
    results = []
    for pdf in pdfs:
        metadata = infer_mock_exam_metadata(pdf, root)
        result = service.process_pdf(
            str(pdf),
            metadata_override=metadata,
            **splitter_config,
        )
        results.append({
            "filename": pdf.name,
            "relative_path": str(pdf.relative_to(root)),
            **metadata,
            **result,
        })
        print(f"{result.get('status')} {pdf.name} chunks={result.get('chunks_count', 0)}")
        partial_report = {
            **summarize_ingestion_results(results),
            "root": str(root),
            "offset": offset,
            "splitter": splitter_config,
            "results": results,
            "complete": False,
        }
        args.report.write_text(json.dumps(partial_report, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        **summarize_ingestion_results(results),
        "root": str(root),
        "offset": offset,
        "splitter": splitter_config,
        "results": results,
        "complete": True,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""교육과정 폴더 전체를 문서 단위로 병렬 OCR하고 검수 보고서를 만든다."""

from __future__ import annotations

import argparse
import json
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ocr_service import OCRService, text_quality


def process_one(args: tuple[str, str, str, int]) -> dict:
    path_text, cache_text, backend, dpi = args
    path = Path(path_text)
    service = OCRService(cache_dir=cache_text, backend=backend, dpi=dpi)
    result = service.extract_pdf(path)
    low_pages = [page.page for page in result.pages if text_quality(page.text) < service.minimum_quality]
    ocr_pages = [page.page for page in result.pages if page.method != "pypdf"]
    return {
        "source": str(path),
        "filename": unicodedata.normalize("NFC", path.name),
        "document_id": result.document_id,
        "pages": len(result.pages),
        "ocr_pages": len(ocr_pages),
        "low_quality_pages": low_pages,
        "methods": result.methods,
        "warnings": result.warnings,
        "status": "success",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("ocr_cache"))
    parser.add_argument("--backend", choices=["tesseract", "easyocr", "auto"], default="tesseract")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--workers", type=int, default=3)
    options = parser.parse_args()

    options.cache.mkdir(parents=True, exist_ok=True)
    service = OCRService(options.cache, backend=options.backend, dpi=options.dpi)
    unique: dict[str, Path] = {}
    duplicates = []
    for path in sorted(options.directory.glob("*.pdf")):
        document_id = service.file_id(path)
        if document_id in unique:
            duplicates.append({"source": str(path), "same_as": str(unique[document_id])})
        else:
            unique[document_id] = path

    jobs = [(str(path), str(options.cache), options.backend, options.dpi) for path in unique.values()]
    results = []
    with ProcessPoolExecutor(max_workers=options.workers) as executor:
        future_map = {executor.submit(process_one, job): job[0] for job in jobs}
        for index, future in enumerate(as_completed(future_map), 1):
            source = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"source": source, "status": "error", "error": str(exc)}
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {Path(source).name}: "
                f"{result.get('status')} OCR={result.get('ocr_pages', 0)}",
                flush=True,
            )

    report = {
        "backend": options.backend,
        "dpi": options.dpi,
        "documents": len(jobs),
        "duplicates": duplicates,
        "results": sorted(results, key=lambda item: item.get("filename", item["source"])),
    }
    (options.cache / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = sum(result["status"] != "success" for result in results)
    print(f"완료: 성공 {len(results) - failed}, 실패 {failed}, 중복 {len(duplicates)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

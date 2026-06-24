#!/usr/bin/env python3
"""OCR 캐시·기출·성취기준 대응표로 로컬 FAISS 지식베이스를 재구축한다."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_service import LangChainService, LocalHashEmbeddings
from ocr_service import OCRService


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curricula", type=Path, default=Path("parsed_exams_native/교육과정"))
    parser.add_argument("--exams", type=Path, default=Path("parsed_exams_native"))
    parser.add_argument("--mapping", type=Path, default=Path("curriculum_mapping/mappings.json"))
    parser.add_argument("--cache", type=Path, default=Path("ocr_cache"))
    parser.add_argument("--output", type=Path, default=Path("faiss_curriculum"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않습니다: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    ocr = OCRService(args.cache, backend="tesseract", dpi=250)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=160,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    documents = []
    source_ids = set()
    for path in sorted(args.curricula.glob("*.pdf")):
        document = ocr.extract_pdf(path)
        if document.document_id in source_ids:
            continue
        source_ids.add(document.document_id)
        metadata = LangChainService._document_metadata(str(path), document.document_id)
        pages = [
            Document(
                page_content=page.text,
                metadata={
                    **metadata, "page": page.page - 1, "page_number": page.page,
                    "extraction_method": page.method, "ocr_confidence": page.confidence,
                },
            )
            for page in document.pages if page.text.strip()
        ]
        documents.extend(splitter.split_documents(pages))

    for path in sorted(args.exams.glob("20*_교육과정?.json")):
        exam = json.loads(path.read_text(encoding="utf-8"))
        for question in exam.get("questions", []):
            documents.append(
                Document(
                    page_content=question["text"],
                    metadata={
                        "source": exam["source"], "filename": nfc(Path(exam["source"]).name),
                        "document_id": question["id"], "document_type": "exam_question",
                        "curriculum_version": ",".join(question.get("curriculum_mentions", [])) or None,
                        "subject": question.get("subject"), "grade_band": None,
                        "exam_year": exam["year"], "form": exam["form"],
                        "question_number": question["number"], "page_number": question["pages"][0],
                    },
                )
            )

    if args.mapping.exists():
        mappings = json.loads(args.mapping.read_text(encoding="utf-8"))
        for row in mappings:
            content = (
                f"2015 성취기준 {row.get('code_2015')}: {row.get('text_2015') or '없음'}\n"
                f"2022 성취기준 {row.get('code_2022')}: {row.get('text_2022') or '없음'}\n"
                f"변화 유형: {row['change_type']} · 유사도: {row['similarity']}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(args.mapping), "filename": args.mapping.name,
                        "document_id": row["mapping_id"], "document_type": "curriculum_mapping",
                        "curriculum_version": "2015↔2022", "subject": row["subject"],
                        "grade_band": row["grade_band"], "change_type": row["change_type"],
                        "review_required": row["review_required"],
                    },
                )
            )

    if not documents:
        raise SystemExit("색인할 문서가 없습니다")
    embeddings = LocalHashEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(args.output), index_name="index")
    manifest = {
        "embedding_provider": "local_hash", "documents": len(documents),
        "source_document_ids": sorted(source_ids),
    }
    (args.output / "documents.json").write_text(
        json.dumps({"document_ids": sorted(source_ids)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "build_report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LangGraph workflow for PDF upload/OCR/index ingestion."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph_runtime import get_langgraph_runtime, new_thread_id
from ocr_service import OCRDocument, OCRPage, text_quality

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency guard
    END = START = StateGraph = None


class DocumentIngestionState(TypedDict, total=False):
    pdf_path: str
    thread_id: str
    metadata_override: dict[str, Any]
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    extracted: dict[str, Any]
    metadata: dict[str, Any]
    quality_report: dict[str, Any]
    page_documents: list[Document]
    chunks: list[Document]
    result: dict[str, Any]
    trace: list[dict[str, Any]]


def run_pdf_ingestion_graph(
    service: Any,
    pdf_path: str,
    metadata_override: dict[str, Any] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    run_thread_id = thread_id or new_thread_id("document-ingest")
    initial: DocumentIngestionState = {
        "pdf_path": pdf_path,
        "thread_id": run_thread_id,
        "metadata_override": metadata_override or {},
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators or ["\n\n", "\n", "。", ". ", " ", ""],
        "trace": [],
    }
    if StateGraph is None:
        return _run_linear(service, initial)

    runtime = get_langgraph_runtime()
    builder = StateGraph(DocumentIngestionState)

    def extract(state: DocumentIngestionState) -> dict[str, Any]:
        extracted = service.ocr_service.extract_pdf(state["pdf_path"])
        return {
            "extracted": extracted.to_dict(),
            "trace": _append_trace(
                state,
                "extract_ocr",
                "ok",
                {
                    "pages_count": len(extracted.pages),
                    "methods": extracted.methods,
                    "warnings_count": len(extracted.warnings),
                },
            ),
        }

    def classify(state: DocumentIngestionState) -> dict[str, Any]:
        extracted = _ocr_document_from_dict(state["extracted"])
        metadata = service._document_metadata(state["pdf_path"], extracted.document_id)
        metadata.update(state.get("metadata_override", {}))
        return {
            "metadata": metadata,
            "trace": _append_trace(
                state,
                "classify",
                "ok",
                {
                    "document_type": metadata.get("document_type"),
                    "subject": metadata.get("subject"),
                    "collection": metadata.get("collection"),
                },
            ),
        }

    def quality_check(state: DocumentIngestionState) -> dict[str, Any]:
        extracted = _ocr_document_from_dict(state["extracted"])
        report = _quality_report(extracted)
        return {
            "quality_report": report,
            "trace": _append_trace(
                state,
                "quality_check",
                "ok" if report["usable_pages"] else "warning",
                report,
            ),
        }

    def split(state: DocumentIngestionState) -> dict[str, Any]:
        extracted = _ocr_document_from_dict(state["extracted"])
        if extracted.document_id in service.indexed_document_ids:
            return {
                "page_documents": [],
                "chunks": [],
                "trace": _append_trace(
                    state,
                    "split",
                    "skipped",
                    {"reason": "duplicate_document"},
                ),
            }
        documents = _page_documents(extracted, state["metadata"])
        if not documents:
            raise ValueError("PDF에서 검색 가능한 텍스트를 추출하지 못했습니다")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=state["chunk_size"],
            chunk_overlap=state["chunk_overlap"],
            separators=state["separators"],
        )
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_number"] = index
            chunk.metadata["chunk_size"] = state["chunk_size"]
            chunk.metadata["chunk_overlap"] = state["chunk_overlap"]
        return {
            "page_documents": documents,
            "chunks": chunks,
            "trace": _append_trace(
                state,
                "split",
                "ok",
                {
                    "page_documents": len(documents),
                    "chunks_count": len(chunks),
                    "chunk_size": state["chunk_size"],
                    "chunk_overlap": state["chunk_overlap"],
                },
            ),
        }

    def index(state: DocumentIngestionState) -> dict[str, Any]:
        extracted = _ocr_document_from_dict(state["extracted"])
        metadata = state["metadata"]
        if extracted.document_id in service.indexed_document_ids:
            result = _result(
                status="success",
                message="이미 색인된 동일 문서입니다.",
                extracted=extracted,
                metadata=metadata,
                chunks_count=0,
                indexed=False,
                quality_report=state["quality_report"],
                thread_id=state["thread_id"],
                trace=_append_trace(state, "index", "skipped", {"reason": "duplicate_document"}),
            )
            return {"result": result, "trace": result["trace"]}
        service._index_chunks(state["chunks"], extracted.document_id)
        result = _result(
            status="success",
            message=f"PDF 처리 완료: {len(state['chunks'])}개 청크 생성",
            extracted=extracted,
            metadata=metadata,
            chunks_count=len(state["chunks"]),
            indexed=True,
            quality_report=state["quality_report"],
            thread_id=state["thread_id"],
            trace=_append_trace(
                state,
                "index",
                "ok",
                {"chunks_count": len(state["chunks"])},
            ),
        )
        return {"result": result, "trace": result["trace"]}

    builder.add_node("extract_ocr", extract)
    builder.add_node("classify", classify)
    builder.add_node("quality_check", quality_check)
    builder.add_node("split", split)
    builder.add_node("index", index)
    builder.add_edge(START, "extract_ocr")
    builder.add_edge("extract_ocr", "classify")
    builder.add_edge("classify", "quality_check")
    builder.add_edge("quality_check", "split")
    builder.add_edge("split", "index")
    builder.add_edge("index", END)
    graph = runtime.compile(builder)
    config = runtime.config(run_thread_id) if runtime.available else None
    final_state = graph.invoke(initial, config=config) if config else graph.invoke(initial)
    return final_state["result"]


def _run_linear(service: Any, state: DocumentIngestionState) -> dict[str, Any]:
    extracted = service.ocr_service.extract_pdf(state["pdf_path"])
    metadata = service._document_metadata(state["pdf_path"], extracted.document_id)
    metadata.update(state.get("metadata_override", {}))
    quality = _quality_report(extracted)
    documents = _page_documents(extracted, metadata)
    if not documents:
        raise ValueError("PDF에서 검색 가능한 텍스트를 추출하지 못했습니다")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=state["chunk_size"],
        chunk_overlap=state["chunk_overlap"],
        separators=state["separators"],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_number"] = index
        chunk.metadata["chunk_size"] = state["chunk_size"]
        chunk.metadata["chunk_overlap"] = state["chunk_overlap"]
    trace = [
        {"node": "linear_extract_classify_quality_split", "status": "ok", "detail": quality}
    ]
    if extracted.document_id in service.indexed_document_ids:
        return _result(
            "success",
            "이미 색인된 동일 문서입니다.",
            extracted,
            metadata,
            0,
            False,
            quality,
            state["thread_id"],
            [*trace, {"node": "index", "status": "skipped", "detail": {"reason": "duplicate_document"}}],
        )
    service._index_chunks(chunks, extracted.document_id)
    return _result(
        "success",
        f"PDF 처리 완료: {len(chunks)}개 청크 생성",
        extracted,
        metadata,
        len(chunks),
        True,
        quality,
        state["thread_id"],
        [*trace, {"node": "index", "status": "ok", "detail": {"chunks_count": len(chunks)}}],
    )


def _append_trace(
    state: DocumentIngestionState,
    node: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        *state.get("trace", []),
        {"node": node, "status": status, "detail": detail or {}},
    ]


def _ocr_document_from_dict(data: dict[str, Any]) -> OCRDocument:
    return OCRDocument(
        document_id=data["document_id"],
        source=data["source"],
        pages=[OCRPage(**page) for page in data["pages"]],
        warnings=data.get("warnings", []),
    )


def _page_documents(extracted: OCRDocument, metadata: dict[str, Any]) -> list[Document]:
    return [
        Document(
            page_content=page.text,
            metadata={
                **metadata,
                "page": page.page - 1,
                "page_number": page.page,
                "extraction_method": page.method,
                "ocr_confidence": page.confidence,
                "text_quality": text_quality(page.text),
            },
        )
        for page in extracted.pages
        if page.text.strip()
    ]


def _quality_report(extracted: OCRDocument) -> dict[str, Any]:
    page_scores = [text_quality(page.text) for page in extracted.pages]
    low_quality_pages = [
        page.page
        for page, score in zip(extracted.pages, page_scores, strict=False)
        if score < 0.45
    ]
    confidences = [
        page.confidence
        for page in extracted.pages
        if page.confidence is not None
    ]
    return {
        "pages_count": len(extracted.pages),
        "usable_pages": sum(1 for page in extracted.pages if page.text.strip()),
        "low_quality_pages": low_quality_pages[:50],
        "low_quality_count": len(low_quality_pages),
        "average_text_quality": round(sum(page_scores) / len(page_scores), 4) if page_scores else 0.0,
        "average_ocr_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "extraction_methods": extracted.methods,
        "warnings": extracted.warnings,
    }


def _result(
    status: str,
    message: str,
    extracted: OCRDocument,
    metadata: dict[str, Any],
    chunks_count: int,
    indexed: bool,
    quality_report: dict[str, Any],
    thread_id: str,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "chunks_count": chunks_count,
        "pages_count": len(extracted.pages),
        "document_id": extracted.document_id,
        "extraction_methods": extracted.methods,
        "warnings": extracted.warnings,
        "indexed": indexed,
        "document_type": metadata.get("document_type"),
        "quality_report": quality_report,
        "thread_id": thread_id,
        "trace": trace,
        "workflow": "langgraph_pdf_ingestion",
    }

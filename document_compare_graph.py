"""LangGraph workflow for balanced multi-document comparison."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph_runtime import get_langgraph_runtime, new_thread_id

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency guard
    END = START = StateGraph = None


class DocumentComparisonState(TypedDict, total=False):
    question: str
    document_ids: list[str]
    thread_id: str
    documents_by_id: dict[str, list[Any]]
    document_summaries: list[dict[str, Any]]
    answer: str
    documents: list[dict[str, Any]]
    missing_document_ids: list[str]
    trace: list[dict[str, Any]]
    workflow: str


def _append_trace(
    state: DocumentComparisonState,
    node: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        *state.get("trace", []),
        {
            "node": node,
            "status": status,
            "detail": detail or {},
        },
    ]


def run_document_comparison_graph(
    service: Any,
    question: str,
    document_ids: list[str],
    k_per_doc: int = 6,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run a small stateful workflow for document comparison.

    The graph keeps retrieval, per-document summarization, and synthesis as
    explicit steps so later versions can add human review, retries, or
    checkpointing without rewriting the comparison feature.
    """
    clean_ids = service._normalize_document_ids(None, document_ids)
    run_thread_id = thread_id or new_thread_id("document-compare")
    initial: DocumentComparisonState = {
        "question": question,
        "document_ids": clean_ids,
        "thread_id": run_thread_id,
        "trace": [],
    }
    if StateGraph is None:
        return _run_linear(service, initial, k_per_doc)

    runtime = get_langgraph_runtime()
    builder = StateGraph(DocumentComparisonState)

    def retrieve(state: DocumentComparisonState) -> dict[str, Any]:
        documents_by_id = service._retrieve_documents_by_ids(
            state["question"],
            state["document_ids"],
            k_per_doc=k_per_doc,
        )
        missing = [
            document_id
            for document_id in state["document_ids"]
            if not documents_by_id.get(document_id)
        ]
        return {
            "documents_by_id": documents_by_id,
            "missing_document_ids": missing,
            "trace": _append_trace(
                state,
                "retrieve",
                "ok",
                {
                    "document_count": len(documents_by_id),
                    "missing_count": len(missing),
                    "k_per_doc": k_per_doc,
                },
            ),
        }

    def summarize(state: DocumentComparisonState) -> dict[str, Any]:
        summaries = service._summarize_documents_by_id(
            state["question"],
            state["documents_by_id"],
        )
        return {
            "document_summaries": summaries,
            "trace": _append_trace(
                state,
                "summarize",
                "ok",
                {"summary_count": len(summaries)},
            ),
        }

    def synthesize(state: DocumentComparisonState) -> dict[str, Any]:
        answer = service._synthesize_document_comparison(
            state["question"],
            state["document_summaries"],
            state["documents_by_id"],
            state.get("missing_document_ids", []),
        )
        return {
            "answer": answer,
            "documents": service._format_retrieved_documents(
                service._flatten_documents_by_id(state["documents_by_id"])
            ),
            "trace": _append_trace(
                state,
                "synthesize",
                "ok",
                {"answer_chars": len(answer)},
            ),
            "workflow": "langgraph",
        }

    builder.add_node("retrieve", retrieve)
    builder.add_node("summarize", summarize)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "summarize")
    builder.add_edge("summarize", "synthesize")
    builder.add_edge("synthesize", END)
    graph = runtime.compile(builder)
    config = runtime.config(run_thread_id) if runtime.available else None
    if config:
        return graph.invoke(initial, config=config)
    return graph.invoke(initial)


def _run_linear(
    service: Any,
    state: DocumentComparisonState,
    k_per_doc: int,
) -> dict[str, Any]:
    documents_by_id = service._retrieve_documents_by_ids(
        state["question"],
        state["document_ids"],
        k_per_doc=k_per_doc,
    )
    missing = [
        document_id
        for document_id in state["document_ids"]
        if not documents_by_id.get(document_id)
    ]
    summaries = service._summarize_documents_by_id(state["question"], documents_by_id)
    answer = service._synthesize_document_comparison(
        state["question"],
        summaries,
        documents_by_id,
        missing,
    )
    return {
        **state,
        "documents_by_id": documents_by_id,
        "document_summaries": summaries,
        "answer": answer,
        "documents": service._format_retrieved_documents(
            service._flatten_documents_by_id(documents_by_id)
        ),
        "missing_document_ids": missing,
        "trace": [
            {
                "node": "linear_fallback",
                "status": "ok",
                "detail": {
                    "document_count": len(documents_by_id),
                    "summary_count": len(summaries),
                },
            }
        ],
        "workflow": "linear_fallback",
    }

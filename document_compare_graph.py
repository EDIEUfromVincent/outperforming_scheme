"""LangGraph workflow for balanced multi-document comparison."""

from __future__ import annotations

from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency guard
    END = START = StateGraph = None


class DocumentComparisonState(TypedDict, total=False):
    question: str
    document_ids: list[str]
    documents_by_id: dict[str, list[Any]]
    document_summaries: list[dict[str, Any]]
    answer: str
    documents: list[dict[str, Any]]
    missing_document_ids: list[str]
    workflow: str


def run_document_comparison_graph(
    service: Any,
    question: str,
    document_ids: list[str],
    k_per_doc: int = 6,
) -> dict[str, Any]:
    """Run a small stateful workflow for document comparison.

    The graph keeps retrieval, per-document summarization, and synthesis as
    explicit steps so later versions can add human review, retries, or
    checkpointing without rewriting the comparison feature.
    """
    clean_ids = service._normalize_document_ids(None, document_ids)
    initial: DocumentComparisonState = {
        "question": question,
        "document_ids": clean_ids,
    }
    if StateGraph is None:
        return _run_linear(service, initial, k_per_doc)

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
        }

    def summarize(state: DocumentComparisonState) -> dict[str, Any]:
        return {
            "document_summaries": service._summarize_documents_by_id(
                state["question"],
                state["documents_by_id"],
            )
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
            "workflow": "langgraph",
        }

    builder.add_node("retrieve", retrieve)
    builder.add_node("summarize", summarize)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "summarize")
    builder.add_edge("summarize", "synthesize")
    builder.add_edge("synthesize", END)
    graph = builder.compile()
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
        "workflow": "linear_fallback",
    }

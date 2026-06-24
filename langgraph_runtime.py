"""Shared LangGraph runtime helpers.

This module keeps graph persistence wiring in one place. The current app uses
in-memory checkpointers for local development; production should swap this for
a database-backed checkpointer without changing individual graph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from typing import Any

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
except Exception as exc:  # pragma: no cover - optional dependency guard
    InMemorySaver = None
    InMemoryStore = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class LangGraphRuntime:
    checkpointer: Any = None
    store: Any = None
    mode: str = "unavailable"
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.checkpointer is not None

    def config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def compile(self, builder: Any) -> Any:
        if not self.available:
            return builder.compile()
        try:
            return builder.compile(checkpointer=self.checkpointer, store=self.store)
        except TypeError:
            return builder.compile(checkpointer=self.checkpointer)


_RUNTIME: LangGraphRuntime | None = None


def get_langgraph_runtime() -> LangGraphRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if InMemorySaver is None:
        _RUNTIME = LangGraphRuntime(
            mode="unavailable",
            unavailable_reason=str(_IMPORT_ERROR),
        )
        return _RUNTIME
    _RUNTIME = LangGraphRuntime(
        checkpointer=InMemorySaver(),
        store=InMemoryStore() if InMemoryStore is not None else None,
        mode="memory",
    )
    return _RUNTIME


def new_thread_id(prefix: str) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in prefix)
    return f"{safe_prefix}-{uuid4().hex}"


def langgraph_status() -> dict[str, Any]:
    runtime = get_langgraph_runtime()
    return {
        "available": runtime.available,
        "persistence_mode": runtime.mode,
        "has_checkpointer": runtime.checkpointer is not None,
        "has_store": runtime.store is not None,
        "unavailable_reason": runtime.unavailable_reason,
    }

"""2015↔2022 성취기준 대응표와 관련 기출을 검색하는 서비스."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS

from langchain_service import LocalHashEmbeddings


class CurriculumComparisonService:
    def __init__(
        self,
        mapping_path: str | Path = "curriculum_mapping/mappings.json",
        vector_db_path: str | Path = "faiss_curriculum",
    ):
        self.mapping_path = Path(mapping_path)
        self.vector_db_path = Path(vector_db_path)
        self.embeddings = LocalHashEmbeddings()
        self.mappings = self._load_mappings()
        self.vector_store = self._load_vector_store()

    def refresh(self) -> None:
        self.mappings = self._load_mappings()
        self.vector_store = self._load_vector_store()

    def _load_mappings(self) -> list[dict]:
        try:
            return json.loads(self.mapping_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _load_vector_store(self):
        if not (self.vector_db_path / "index.faiss").exists():
            return None
        return FAISS.load_local(
            str(self.vector_db_path), self.embeddings,
            index_name="index", allow_dangerous_deserialization=True,
        )

    def filters(self) -> dict:
        return {
            "subjects": sorted({row["subject"] for row in self.mappings}),
            "grade_bands": sorted({row["grade_band"] for row in self.mappings}),
            "domains": sorted(
                {
                    code for row in self.mappings
                    for code in [row.get("domain_code_2015"), row.get("domain_code_2022")]
                    if code
                }
            ),
            "change_types": sorted({row["change_type"] for row in self.mappings}),
        }

    def compare(
        self,
        subject: str | None = None,
        grade_band: str | None = None,
        domain: str | None = None,
        query: str = "",
        limit: int = 10,
    ) -> dict:
        rows = [
            row for row in self.mappings
            if (not subject or row["subject"] == subject)
            and (not grade_band or row["grade_band"] == grade_band)
            and (
                not domain
                or row.get("domain_code_2015") == domain
                or row.get("domain_code_2022") == domain
            )
        ]
        if query.strip() and rows:
            expanded_query = self._expand_query(query)
            query_vector = np.asarray(self.embeddings.embed_query(expanded_query), dtype=np.float32)
            row_vectors = np.asarray(
                self.embeddings.embed_documents(
                    [f"{row.get('text_2015') or ''} {row.get('text_2022') or ''}" for row in rows]
                ),
                dtype=np.float32,
            )
            scores = np.nan_to_num(
                np.sum(row_vectors * query_vector, axis=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            ranked = sorted(zip(scores.tolist(), rows), key=lambda item: item[0], reverse=True)
            rows = [{**row, "query_score": round(score, 4)} for score, row in ranked]
        rows = rows[:limit]
        related_exams = self._related_exams(query or self._comparison_query(rows), limit=6)
        return {
            "count": len(rows), "comparisons": rows,
            "related_exams": related_exams,
            "notice": "대응표의 semantic_match 및 unmatched 항목은 자동 추정이므로 검수가 필요합니다.",
        }

    def _related_exams(self, query: str, limit: int) -> list[dict]:
        if self.vector_store is None or not query.strip():
            return []
        documents = self.vector_store.similarity_search(
            query, k=limit, filter={"document_type": "exam_question"}
        )
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

    @staticmethod
    def _expand_query(query: str) -> str:
        """수험생식 짧은 명사형 질문을 교육과정 문장형 표현과 가볍게 연결한다."""
        expansions = {
            "읽기 쓰기": "읽기 쓰기 읽고 쓰기 읽고 쓸 수 있다",
            "듣고 말하기": "듣고 말하기 듣기 말하기 듣고 말할 수 있다",
            "토의 토론": "토의 토론 토의·토론 토론하다",
            "과정 중심 평가": "과정 중심 평가 과정중심평가 평가 피드백",
            "분수의 나눗셈": "분수의 나눗셈 나누기 계산 원리",
        }
        additions = [value for key, value in expansions.items() if key in query]
        return " ".join([query, *additions])

    @staticmethod
    def _comparison_query(rows: list[dict]) -> str:
        return " ".join(
            f"{row.get('text_2015') or ''} {row.get('text_2022') or ''}"
            for row in rows[:3]
        )[:3000]

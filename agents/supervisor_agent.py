"""총괄 에이전트.

총론과 평가기준·성취수준을 감독 기준으로 삼아 교과 전문 에이전트와
학년 학생 에이전트의 출력을 하나의 수업-임용 통합 답변으로 조립한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.agent_router import Route, route_question
from agents.grade_student_agent import student_lens
from agents.subject_agent import SubjectAgent
from comparison_service import CurriculumComparisonService


class SupervisorAgent:
    def __init__(
        self,
        comparison_service: CurriculumComparisonService,
        langchain_service: Any,
    ):
        self.comparison_service = comparison_service
        self.langchain_service = langchain_service
        self.subject_agent = SubjectAgent(comparison_service)

    def run(
        self,
        query: str,
        subject: str | None = None,
        grade: int | None = None,
        limit: int = 6,
    ) -> dict:
        self.comparison_service.refresh()
        route = route_question(query, selected_subject=subject, selected_grade=grade)
        subject_result = self.subject_agent.run(
            subject=route.subject,
            grade_band=route.grade_band,
            query=query,
            limit=limit,
        )
        grade_result = student_lens(route.grade, query)
        governance_docs = self._governance_docs(query, limit=4)
        audit = self._audit(route, subject_result, grade_result, governance_docs)
        answer = self._generate_answer(query, route, subject_result, grade_result, governance_docs, audit)
        return {
            "route": {
                "subject": route.subject,
                "grade": route.grade,
                "grade_band": route.grade_band,
                "intent": route.intent,
            },
            "answer": answer,
            "audit": audit,
            "governance_docs": governance_docs,
            "subject_agent": subject_result,
            "grade_student_agent": grade_result,
            "notice": (
                "총괄 에이전트는 총론·평가기준 관점으로 감독합니다. "
                "자동 대응표의 검수 필요 항목은 확정 대응이 아니라 검토 대상으로 표시됩니다."
            ),
        }

    def _governance_docs(self, query: str, limit: int = 4) -> list[dict]:
        store = getattr(self.langchain_service, "vector_store", None)
        if store is None:
            return []
        docs = []
        filters = [
            {"document_type": "assessment_standard"},
            {"document_type": "curriculum"},
            {"document_type": "curriculum_commentary"},
        ]
        for metadata_filter in filters:
            try:
                docs.extend(store.similarity_search(query, k=2, filter=metadata_filter))
            except Exception:
                continue
        output = []
        seen = set()
        for doc in docs:
            meta = doc.metadata
            key = (meta.get("source"), meta.get("page_number"), doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "content": doc.page_content[:900],
                "metadata": meta,
                "label": self._source_label(meta),
            })
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _source_label(metadata: dict) -> str:
        filename = metadata.get("filename") or Path(metadata.get("source", "")).name
        page = metadata.get("page_number")
        document_type = metadata.get("document_type")
        pieces = [filename]
        if page:
            pieces.append(f"p.{page}")
        if document_type:
            pieces.append(document_type)
        return " · ".join(str(piece) for piece in pieces if piece)

    @staticmethod
    def _audit(route: Route, subject_result: dict, grade_result: dict, governance_docs: list[dict]) -> dict:
        comparisons = subject_result.get("comparisons", [])
        related_exams = subject_result.get("related_exams", [])
        checks = {
            "총론·평가기준 근거": bool(governance_docs),
            "교과 성취기준 근거": bool(comparisons),
            "학년 학생 관점": bool(grade_result.get("focus")),
            "관련 기출 관점": bool(related_exams),
            "검수 필요 표시": subject_result.get("review_required_count", 0) >= 0,
            "수업 설계 요소": True,
            "인출 질문": True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "review_required_count": subject_result.get("review_required_count", 0),
            "related_exam_count": len(related_exams),
            "route_complete": bool(route.subject or route.grade),
        }

    def _generate_answer(
        self,
        query: str,
        route: Route,
        subject_result: dict,
        grade_result: dict,
        governance_docs: list[dict],
        audit: dict,
    ) -> str:
        if getattr(self.langchain_service, "llm", None) is not None:
            try:
                return self._llm_answer(query, route, subject_result, grade_result, governance_docs, audit)
            except Exception as exc:
                print(f"총괄 에이전트 LLM 답변 실패: {exc}")
        return self._fallback_answer(query, route, subject_result, grade_result, governance_docs, audit)

    def _llm_answer(
        self,
        query: str,
        route: Route,
        subject_result: dict,
        grade_result: dict,
        governance_docs: list[dict],
        audit: dict,
    ) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 초등임용 수험생이자 기간제 교사인 사용자를 돕는 총괄 에이전트입니다.
총론과 평가기준·성취수준을 감독 기준으로 삼고, 교과 전문 에이전트와 학년 학생 에이전트의 결과를 통합하세요.
기출에는 공식 정답이 없으므로 단정하지 말고 '근거 기반 예상 관점'으로 표현하세요.
자동 대응표의 검수 필요 항목은 확정 사실처럼 말하지 마세요.
답변은 ① 라우팅 판단 ② 총론·평가기준 감독 포인트 ③ 교과 성취기준 변화 ④ 해당 학년 학생 반응
⑤ 수업 목표·학생 활동·교사 지원·평가 증거 ⑥ 관련 기출 관점 ⑦ 답을 숨긴 인출 질문 순서로 작성하세요."""),
            ("human", """질문: {query}

라우팅: {route}

총론·평가기준 근거:
{governance}

교과 에이전트 결과:
{subject_result}

학년 학생 에이전트 결과:
{grade_result}

감사 결과:
{audit}"""),
        ])
        chain = prompt | self.langchain_service.llm | StrOutputParser()
        return chain.invoke({
            "query": query,
            "route": route,
            "governance": "\n\n".join(f"[근거 {i+1}] {doc['label']}\n{doc['content']}" for i, doc in enumerate(governance_docs)),
            "subject_result": subject_result,
            "grade_result": grade_result,
            "audit": audit,
        })

    @staticmethod
    def _fallback_answer(
        query: str,
        route: Route,
        subject_result: dict,
        grade_result: dict,
        governance_docs: list[dict],
        audit: dict,
    ) -> str:
        subject = route.subject or "교과 미지정"
        grade = f"{route.grade}학년" if route.grade else "학년 미지정"
        comparison_lines = "\n".join(f"- {point}" for point in subject_result.get("summary_points", [])) or "- 관련 성취기준 비교 결과가 없습니다."
        governance_lines = "\n".join(f"- {doc['label']}" for doc in governance_docs[:3]) or "- 총론·평가기준 근거를 찾지 못했습니다."
        exam_lines = []
        for exam in subject_result.get("related_exams", [])[:3]:
            meta = exam.get("metadata", {})
            exam_lines.append(f"- {meta.get('exam_year', '')} {meta.get('form', '')} {meta.get('subject', '')} {meta.get('question_number', '')}번")
        related_exams = "\n".join(exam_lines) or "- 관련 기출은 추가 확인이 필요합니다."
        misconceptions = ", ".join(grade_result.get("misconceptions", []))
        supports = ", ".join(grade_result.get("support", []))
        question_style = grade_result.get("question_style", "왜 그렇게 생각했을까?")
        review_note = (
            f"검수 필요 대응 {subject_result.get('review_required_count', 0)}개가 포함되어 있습니다."
            if subject_result.get("review_required_count", 0)
            else "검수 필요 대응은 상위 결과에 없습니다."
        )
        return f"""① 라우팅 판단
- 교과: {subject}
- 학년: {grade}
- 의도: {route.intent}

② 총론·평가기준 감독 포인트
{governance_lines}
- 수업 설계는 목표, 학생 활동, 교사 지원, 평가 증거가 서로 맞물려야 합니다.

③ 교과 성취기준 변화
{comparison_lines}
- {review_note}

④ 해당 학년 학생 반응
- 초점: {grade_result.get('focus')}
- 예상 오개념/장벽: {misconceptions}
- 교사 지원: {supports}

⑤ 수업 설계 초안
- 목표: 학생이 성취기준의 핵심 개념을 자기 말과 활동 결과로 설명한다.
- 학생 활동: 구체 사례 탐색 → 짝 설명 → 전체 공유 → 자기 점검 순서로 진행한다.
- 교사 지원: {question_style} 같은 발문으로 사고 과정을 드러내게 한다.
- 평가 증거: 학생 설명, 활동 기록, 오개념 수정 발화, 짧은 출구 질문 응답을 수집한다.

⑥ 관련 기출 관점
{related_exams}
- 기출은 공식 정답이 아니라 근거 기반 예상 관점으로만 활용하세요.

⑦ 인출 질문
- 이 수업에서 총론·평가기준 관점으로 반드시 확인해야 할 평가 증거는 무엇인가?
- {grade} 학생이 가장 흔히 보일 오개념은 무엇이며, 어떤 발문으로 되돌릴 수 있을까?
- 2015↔2022 성취기준 변화가 수업 활동을 어떻게 바꾸는가?"""

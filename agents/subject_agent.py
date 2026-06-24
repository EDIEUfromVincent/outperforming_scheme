"""교과 전문 에이전트.

현재 구축된 2015↔2022 대응표와 관련 기출 검색을 교과·학년군 단위로
호출하고, 수업 설계에 쓸 수 있는 핵심 근거를 구조화한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from comparison_service import CurriculumComparisonService


CHANGE_LABELS = {
    "unchanged": "유지",
    "same_code_modified": "같은 코드, 문구 수정",
    "semantic_match": "의미상 대응 추정",
    "removed_or_unmatched": "2015에는 있으나 2022 직접 대응 미확인",
    "new_or_unmatched": "2022 신설 또는 2015 직접 대응 미확인",
}


@dataclass
class SubjectAgentResult:
    subject: str | None
    grade_band: str | None
    query: str
    comparisons: list[dict]
    related_exams: list[dict]
    review_required_count: int
    summary_points: list[str]


class SubjectAgent:
    def __init__(self, comparison_service: CurriculumComparisonService):
        self.comparison_service = comparison_service

    def run(
        self,
        subject: str | None,
        grade_band: str | None,
        query: str,
        limit: int = 6,
    ) -> dict:
        result = self.comparison_service.compare(
            subject=subject,
            grade_band=grade_band,
            query=query,
            limit=limit,
        )
        rows = result.get("comparisons", [])
        related_exams = result.get("related_exams", [])
        if not related_exams and rows:
            fallback_query = " ".join(
                f"{row.get('text_2015') or ''} {row.get('text_2022') or ''}"
                for row in rows[:3]
            )[:2500]
            related_exams = self.comparison_service._related_exams(fallback_query, limit=6)
        summary_points = []
        for row in rows[:4]:
            label = CHANGE_LABELS.get(row["change_type"], row["change_type"])
            left = row.get("code_2015") or "2015 대응 없음"
            right = row.get("code_2022") or "2022 대응 없음"
            review = " · 검수 필요" if row.get("review_required") else ""
            summary_points.append(f"{row['subject']} {row['grade_band']} {left} → {right}: {label}{review}")
        return asdict(
            SubjectAgentResult(
                subject=subject,
                grade_band=grade_band,
                query=query,
                comparisons=rows,
                related_exams=related_exams,
                review_required_count=sum(1 for row in rows if row.get("review_required")),
                summary_points=summary_points,
            )
        )

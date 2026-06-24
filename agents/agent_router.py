"""교과·학년 라우팅 유틸리티.

질문 속 단서와 사용자가 선택한 값을 합쳐 교과 전문 에이전트와
1~6학년 학생 에이전트를 선택한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SUBJECTS = [
    "국어", "수학", "사회", "과학", "도덕", "실과",
    "체육", "음악", "미술", "영어", "통합교과",
    "총론·창의적 체험활동",
]

SUBJECT_ALIASES = {
    "바른 생활": "통합교과",
    "슬기로운 생활": "통합교과",
    "즐거운 생활": "통합교과",
    "창체": "총론·창의적 체험활동",
    "창의적 체험활동": "총론·창의적 체험활동",
    "총론": "총론·창의적 체험활동",
}


@dataclass
class Route:
    subject: str | None
    grade: int | None
    grade_band: str | None
    intent: str
    query: str


def grade_to_band(grade: int | None) -> str | None:
    if grade in {1, 2}:
        return "1-2"
    if grade in {3, 4}:
        return "3-4"
    if grade in {5, 6}:
        return "5-6"
    return None


def infer_subject(text: str, selected_subject: str | None = None) -> str | None:
    if selected_subject and selected_subject != "전체":
        return SUBJECT_ALIASES.get(selected_subject, selected_subject)
    for alias, subject in SUBJECT_ALIASES.items():
        if alias in text:
            return subject
    for subject in SUBJECTS:
        if subject in text:
            return subject
    return None


def infer_grade(text: str, selected_grade: int | None = None) -> int | None:
    if selected_grade in {1, 2, 3, 4, 5, 6}:
        return selected_grade
    match = re.search(r"([1-6])\s*학년", text)
    if match:
        return int(match.group(1))
    return None


def infer_intent(text: str) -> str:
    compact = text.replace(" ", "")
    if any(term in compact for term in ["수업설계", "수업을", "지도안", "발문", "활동"]):
        return "lesson_design"
    if any(term in compact for term in ["인출", "복습", "퀴즈", "문제"]):
        return "retrieval_practice"
    if any(term in compact for term in ["비교", "2015", "2022", "변화"]):
        return "curriculum_comparison"
    return "integrated_coaching"


def route_question(
    query: str,
    selected_subject: str | None = None,
    selected_grade: int | None = None,
) -> Route:
    grade = infer_grade(query, selected_grade)
    return Route(
        subject=infer_subject(query, selected_subject),
        grade=grade,
        grade_band=grade_to_band(grade),
        intent=infer_intent(query),
        query=query,
    )

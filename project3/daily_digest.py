"""초등임용 일일 브리핑 생성.

구성:
- 기출문제 1문제: parsed_exams_native의 실제 기출 JSON에서 선택
- 예비문제 1문제: 성취기준/기출 기반 로컬 템플릿으로 생성
- 오늘의 임용 공지·교육과정 소식: 공식 사이트 수집 결과
"""

from __future__ import annotations

import json
import random
import re
from datetime import date
from pathlib import Path

from notice_crawler import collect_notices


ROOT_DIR = Path(__file__).resolve().parents[1]
EXAM_DIR = ROOT_DIR / "parsed_exams_native"
STANDARDS_PATH = ROOT_DIR / "curriculum_mapping" / "standards.json"
CORE_LINKS = [
    ("한국교육과정평가원", "기출·시험자료", "https://www.kice.re.kr/"),
    ("교육부", "교육과정·정책", "https://www.moe.go.kr/"),
    ("국가교육과정정보센터 2022 개정 교육과정 자료", "교육과정 원문", "https://ncic.re.kr/cntnt/198.do"),
    ("국가교육과정정보센터 2015 개정 교육과정 자료", "교육과정 원문", "https://ncic.re.kr/cntnt/199.do"),
    ("온라인 교직원 채용", "시도교육청 임용 공고", "https://www.edurecruit.go.kr/nxui/index.html"),
]


def build_daily_digest(
    recipient: str = "ohjinwoo9696@gmail.com",
    include_regions: bool = False,
    seed: str | None = None,
) -> dict:
    """메일 제목/본문/원자료를 생성한다."""
    today = date.today().isoformat()
    rng = random.Random(seed or today)
    past_question = pick_exam_question(rng)
    practice_question = build_practice_question(rng, past_question)
    notices = collect_notices(include_regions=include_regions)
    subject = f"[초등임용 일일 브리핑] {today} 기출 1문제 · 예비 1문제 · 교육소식"
    body = render_email_body(
        today=today,
        recipient=recipient,
        past_question=past_question,
        practice_question=practice_question,
        notices=notices,
    )
    return {
        "subject": subject,
        "body": body,
        "recipient": recipient,
        "past_question": past_question,
        "practice_question": practice_question,
        "notices": notices,
    }


def pick_exam_question(rng: random.Random) -> dict:
    files = sorted(
        path for path in EXAM_DIR.glob("*.json")
        if re.match(r"\d{4}_교육과정[AB]\.json", path.name)
    )
    questions = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for question in data.get("questions", []):
            text = _clean(question.get("text", ""))
            if len(text) < 120:
                continue
            questions.append({
                "year": data.get("year"),
                "form": data.get("form"),
                "number": question.get("number"),
                "subject": question.get("subject"),
                "score": question.get("score"),
                "curriculum_mentions": question.get("curriculum_mentions", []),
                "text": text[:1200],
                "source": data.get("source"),
            })
    if not questions:
        return {
            "year": None,
            "form": None,
            "number": None,
            "subject": None,
            "score": None,
            "text": "기출문제 JSON을 찾지 못했습니다.",
        }
    return rng.choice(questions)


def build_practice_question(rng: random.Random, past_question: dict) -> dict:
    standards = _load_standards()
    subject = past_question.get("subject")
    candidates = [
        row for row in standards
        if row.get("version") == "2022"
        and row.get("subject") == subject
        and row.get("code")
        and row.get("text")
    ]
    if not candidates:
        candidates = [
            row for row in standards
            if row.get("version") == "2022" and row.get("code") and row.get("text")
        ]
    standard = rng.choice(candidates) if candidates else {}
    code = standard.get("code", "성취기준 확인 필요")
    standard_text = _clean(standard.get("text", ""))
    practice_text = f"""[예비문제]
다음 성취기준을 바탕으로 초등임용 답안의 뼈대를 구성하시오.

성취기준: {code}
{standard_text or '성취기준 원문을 확인하시오.'}

1) 이 성취기준이 요구하는 학생의 핵심 수행을 한 문장으로 쓰시오.
2) 실제 수업에서 학생 활동 1가지와 교사 발문 1가지를 제시하시오.
3) 평가기준 관점에서 수집할 수 있는 평가 증거 2가지를 쓰시오.
4) 관련 기출문제가 요구할 수 있는 출제 관점을 1가지 예상하시오.

※ 공식 기출 정답이 아니라 인출 훈련용 예비문제입니다."""
    return {
        "standard_code": code,
        "subject": standard.get("subject"),
        "grade_band": standard.get("grade_band"),
        "text": practice_text,
    }


def render_email_body(
    today: str,
    recipient: str,
    past_question: dict,
    practice_question: dict,
    notices: dict,
) -> str:
    notice_lines = []
    for index, item in enumerate(notices.get("items", [])[:12], 1):
        notice_lines.append(
            f"{index}. [{item.get('source')}] {item.get('title')}\n"
            f"   - 분류: {item.get('category')}\n"
            f"   - 링크: {item.get('url')}"
        )
    if not notice_lines:
        notice_lines.append("수집된 새 공지 후보가 없습니다. 공식 사이트 직접 확인이 필요합니다.")
    warning_lines = "\n".join(f"- {warning}" for warning in notices.get("warnings", [])[:8])
    if not warning_lines:
        warning_lines = "- 없음"
    core_link_lines = "\n".join(
        f"- [{name}] {category}: {url}"
        for name, category, url in CORE_LINKS
    )

    return f"""안녕하세요. {recipient} 기준 초등임용 일일 브리핑입니다.

날짜: {today}

==============================
1. 오늘의 기출문제 1문제
==============================
출처: {past_question.get('year')}학년도 교육과정 {past_question.get('form')}형 {past_question.get('number')}번
교과: {past_question.get('subject')} / 배점: {past_question.get('score')}

{past_question.get('text')}

인출 포인트:
- 이 문항이 묻는 성취기준·교수학습·평가 관점을 분리해 보세요.
- 공식 정답이 아니라, 먼저 자신의 답안 구조를 5분 안에 써 보세요.

==============================
2. 오늘의 예비문제 1문제
==============================
교과: {practice_question.get('subject')} / 학년군: {practice_question.get('grade_band')}

{practice_question.get('text')}

==============================
3. 오늘의 임용 공지·교육과정 소식
==============================
{chr(10).join(notice_lines)}

수집 경고:
{warning_lines}

핵심 공식 바로가기:
{core_link_lines}

==============================
4. 오늘의 공부 루프
==============================
1) 기출문제 5분 인출
2) 예비문제 7분 답안 구성
3) 교육뉴스/공지 3분 확인
4) 틀린 부분은 앱의 학습 대시보드에 기록

※ 이 메일은 자동 수집 후보를 포함합니다. 시험 일정·공고·원서접수 등 고위험 정보는 반드시 원문 링크에서 최종 확인하세요.
"""


def _load_standards() -> list[dict]:
    if not STANDARDS_PATH.exists():
        return []
    try:
        return json.loads(STANDARDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    digest = build_daily_digest(include_regions=False)
    print(digest["subject"])
    print(digest["body"])

"""학습 기록, 이동평균선, 취약 성취기준, 에빙하우스 복습 서비스."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from learning_db import DB_PATH, connect, init_db


EBBINGHAUS_INTERVALS = [1, 3, 7, 15, 30]


class LearningService:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        init_db(self.db_path)
        self._standards = self._load_standards()

    @staticmethod
    def _load_standards() -> dict[str, dict]:
        path = Path("curriculum_mapping/standards.json")
        if not path.exists():
            return {}
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        # 2022를 우선하되, 없으면 2015도 사용한다.
        output = {}
        for row in rows:
            code = row.get("code")
            if not code:
                continue
            if code not in output or row.get("version") == "2022":
                output[code] = row
        return output

    def record_attempt(
        self,
        standard_code: str,
        subject: str | None,
        grade_band: str | None,
        is_correct: bool,
        confidence: int = 3,
        time_spent_sec: int = 0,
        question_text: str = "",
        user_answer: str = "",
        source_type: str = "manual",
        created_at: str | None = None,
    ) -> dict:
        created = self._parse_datetime(created_at) if created_at else datetime.now()
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO quiz_attempts (
                    created_at, standard_code, subject, grade_band, source_type,
                    question_text, user_answer, is_correct, confidence, time_spent_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created.isoformat(timespec="seconds"),
                    standard_code,
                    subject,
                    grade_band,
                    source_type,
                    question_text,
                    user_answer,
                    1 if is_correct else 0,
                    int(confidence),
                    int(time_spent_sec),
                ),
            )
            attempt_id = int(cursor.lastrowid)
            for interval in EBBINGHAUS_INTERVALS:
                scheduled = (created.date() + timedelta(days=interval)).isoformat()
                conn.execute(
                    """
                    INSERT INTO review_schedule (
                        attempt_id, standard_code, subject, grade_band,
                        scheduled_at, interval_days, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'due', ?)
                    """,
                    (
                        attempt_id,
                        standard_code,
                        subject,
                        grade_band,
                        scheduled,
                        interval,
                        created.isoformat(timespec="seconds"),
                    ),
                )
        return {
            "attempt_id": attempt_id,
            "standard_code": standard_code,
            "scheduled_reviews": EBBINGHAUS_INTERVALS,
        }

    def save_lecture_note(self, note: dict) -> dict:
        """인강 요약본을 시험 대비 학습자료로 저장한다."""
        created = datetime.now().isoformat(timespec="seconds")
        with connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM lecture_notes WHERE document_id = ?",
                (note["document_id"],),
            ).fetchone()
            if existing:
                return {
                    "lecture_note_id": int(existing["id"]),
                    "document_id": note["document_id"],
                    "indexed": False,
                    "message": "이미 저장된 강의 요약본입니다.",
                }
            cursor = conn.execute(
                """
                INSERT INTO lecture_notes (
                    created_at, document_id, title, lecture_date, subject,
                    topics_json, exam_years_json, knowledge_points_json,
                    assignments_json, raw_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created,
                    note["document_id"],
                    note.get("title") or "강의 요약본",
                    note.get("lecture_date"),
                    note.get("subject"),
                    json.dumps(note.get("topics", []), ensure_ascii=False),
                    json.dumps(note.get("exam_years", []), ensure_ascii=False),
                    json.dumps(note.get("knowledge_points", []), ensure_ascii=False),
                    json.dumps(note.get("assignments", []), ensure_ascii=False),
                    note.get("raw_text", ""),
                ),
            )
        return {
            "lecture_note_id": int(cursor.lastrowid),
            "document_id": note["document_id"],
            "indexed": True,
            "message": "강의 요약본을 학습자료로 저장했습니다.",
        }

    def lecture_notes(self, limit: int = 20, subject: str | None = None) -> list[dict]:
        query = "SELECT * FROM lecture_notes"
        params: list[Any] = []
        if subject:
            query += " WHERE subject = ?"
            params.append(subject)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with connect(self.db_path) as conn:
            rows = [dict(row) for row in conn.execute(query, params)]
        for row in rows:
            for key in ["topics_json", "exam_years_json", "knowledge_points_json", "assignments_json"]:
                try:
                    row[key.removesuffix("_json")] = json.loads(row.pop(key) or "[]")
                except json.JSONDecodeError:
                    row[key.removesuffix("_json")] = []
            row["raw_text"] = row.get("raw_text", "")[:1000]
        return rows

    def daily_metrics(self, days: int = 60, subject: str | None = None) -> list[dict]:
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        query = """
            SELECT substr(created_at, 1, 10) AS day, subject,
                   COUNT(*) AS attempts,
                   SUM(is_correct) AS correct_count,
                   SUM(time_spent_sec) AS study_time_sec
            FROM quiz_attempts
            WHERE substr(created_at, 1, 10) >= ?
        """
        params: list[Any] = [since]
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        query += " GROUP BY day, subject ORDER BY day"
        with connect(self.db_path) as conn:
            rows = [dict(row) for row in conn.execute(query, params)]
        if not rows:
            return []
        df = pd.DataFrame(rows)
        grouped = df.groupby("day", as_index=False).agg(
            attempts=("attempts", "sum"),
            correct_count=("correct_count", "sum"),
            study_time_sec=("study_time_sec", "sum"),
        )
        grouped["accuracy"] = grouped["correct_count"] / grouped["attempts"]

        all_days = pd.DataFrame({
            "day": pd.date_range(start=since, end=date.today().isoformat(), freq="D").strftime("%Y-%m-%d")
        })
        grouped = all_days.merge(grouped, on="day", how="left").fillna({
            "attempts": 0,
            "correct_count": 0,
            "study_time_sec": 0,
        })
        grouped["accuracy"] = grouped["accuracy"].astype(float)
        grouped["ma_5"] = grouped["accuracy"].rolling(5, min_periods=1).mean()
        grouped["ma_20"] = grouped["accuracy"].rolling(20, min_periods=1).mean()
        grouped[["attempts", "correct_count", "study_time_sec"]] = grouped[
            ["attempts", "correct_count", "study_time_sec"]
        ].astype(int)
        return grouped.fillna(0).to_dict(orient="records")

    def aggregate_metrics(self, days: int = 180, period: str = "W", subject: str | None = None) -> list[dict]:
        """일간 기록을 주간/월간으로 묶어 장기 흐름을 반환한다.

        period:
        - "W": 주간
        - "M": 월간
        """
        period = "M" if period.upper().startswith("M") else "W"
        daily = self.daily_metrics(days=days, subject=subject)
        if not daily:
            return []
        df = pd.DataFrame(daily)
        df["day"] = pd.to_datetime(df["day"])
        non_empty = df[df["attempts"] > 0].copy()
        if non_empty.empty:
            return []
        non_empty["period"] = non_empty["day"].dt.to_period(period).astype(str)
        grouped = non_empty.groupby("period", as_index=False).agg(
            attempts=("attempts", "sum"),
            correct_count=("correct_count", "sum"),
            study_time_sec=("study_time_sec", "sum"),
        )
        grouped["accuracy"] = grouped["correct_count"] / grouped["attempts"]
        grouped["ma_5"] = grouped["accuracy"].rolling(5, min_periods=1).mean()
        grouped["ma_20"] = grouped["accuracy"].rolling(20, min_periods=1).mean()
        grouped[["attempts", "correct_count", "study_time_sec"]] = grouped[
            ["attempts", "correct_count", "study_time_sec"]
        ].astype(int)
        return grouped.fillna(0).to_dict(orient="records")

    def weak_standards(self, limit: int = 10) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                """
                SELECT standard_code, subject, grade_band,
                       COUNT(*) AS attempts,
                       SUM(is_correct) AS correct_count,
                       AVG(is_correct) AS accuracy,
                       AVG(confidence) AS avg_confidence,
                       SUM(time_spent_sec) AS study_time_sec,
                       MAX(created_at) AS last_attempt_at
                FROM quiz_attempts
                GROUP BY standard_code, subject, grade_band
                HAVING attempts >= 2
                """
            )]
        output = []
        for row in rows:
            trend = self._standard_trend(row["standard_code"])
            downside_cross = (
                trend.get("ma_5") is not None
                and trend.get("ma_20") is not None
                and trend["ma_5"] < trend["ma_20"]
            )
            weak_score = (1 - float(row["accuracy"] or 0)) * 100
            if downside_cross:
                weak_score += 25
            if float(row.get("avg_confidence") or 3) <= 2:
                weak_score += 10
            standard = self._standards.get(row["standard_code"], {})
            output.append({
                **row,
                "accuracy": round(float(row["accuracy"] or 0), 4),
                "avg_confidence": round(float(row.get("avg_confidence") or 0), 2),
                "weak_score": round(weak_score, 2),
                "downside_cross": downside_cross,
                "ma_5": trend.get("ma_5"),
                "ma_20": trend.get("ma_20"),
                "standard_text": standard.get("text", "")[:500],
            })
        return sorted(output, key=lambda item: item["weak_score"], reverse=True)[:limit]

    def due_reviews(self, target_date: str | None = None, limit: int = 20) -> list[dict]:
        day = target_date or date.today().isoformat()
        with connect(self.db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                """
                SELECT standard_code, subject, grade_band,
                       MIN(scheduled_at) AS next_review_at,
                       COUNT(*) AS due_count,
                       GROUP_CONCAT(interval_days) AS intervals
                FROM review_schedule
                WHERE status = 'due' AND scheduled_at <= ?
                GROUP BY standard_code, subject, grade_band
                ORDER BY next_review_at
                LIMIT ?
                """,
                (day, limit),
            )]
        for row in rows:
            standard = self._standards.get(row["standard_code"], {})
            row["standard_text"] = standard.get("text", "")[:500]
        return rows

    def complete_reviews(self, standard_code: str, completed_at: str | None = None) -> dict:
        """오늘까지 도래한 특정 성취기준 복습 일정을 완료 처리한다."""
        completed_day = (
            self._parse_datetime(completed_at).date().isoformat()
            if completed_at
            else date.today().isoformat()
        )
        with connect(self.db_path) as conn:
            due_count = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM review_schedule
                WHERE status = 'due'
                  AND standard_code = ?
                  AND scheduled_at <= ?
                """,
                (standard_code, completed_day),
            ).fetchone()["count"]
            conn.execute(
                """
                UPDATE review_schedule
                SET status = 'completed'
                WHERE status = 'due'
                  AND standard_code = ?
                  AND scheduled_at <= ?
                """,
                (standard_code, completed_day),
            )
        return {
            "standard_code": standard_code,
            "completed_reviews": int(due_count),
            "completed_at": completed_day,
        }

    def generate_variant(
        self,
        standard_code: str,
        subject: str | None = None,
        grade_band: str | None = None,
        weakness_note: str = "",
    ) -> dict:
        standard = self._standards.get(standard_code, {})
        standard_text = standard.get("text", "")
        subject = subject or standard.get("subject")
        grade_band = grade_band or standard.get("grade_band")
        prompt = f"{subject or ''} {grade_band or ''} {standard_code} {weakness_note}".strip()
        quiz_text = f"""[변형 문제]
성취기준 {standard_code}와 관련하여 학생이 다음과 같은 어려움을 보인다.

- 약점: {weakness_note or '정답은 맞히지만 원리나 평가 증거를 충분히 설명하지 못한다.'}
- 성취기준 근거: {standard_text or '성취기준 원문 확인 필요'}

1) 이 학생의 이해 상태를 진단할 수 있는 교사 발문 2가지를 쓰시오.
2) 학생 활동 중 수집할 평가 증거 2가지를 쓰시오.
3) 위 성취기준을 실제 수업 장면으로 연결하는 짧은 지도 방안을 쓰시오.

※ 기출 공식 답안이 아니라 약점 보완용 근거 기반 변형 문제입니다."""
        created = datetime.now().isoformat(timespec="seconds")
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO generated_quizzes (
                    created_at, standard_code, subject, grade_band, prompt, quiz_text, source
                ) VALUES (?, ?, ?, ?, ?, ?, 'template')
                """,
                (created, standard_code, subject, grade_band, prompt, quiz_text),
            )
            quiz_id = int(cursor.lastrowid)
        return {
            "quiz_id": quiz_id,
            "standard_code": standard_code,
            "subject": subject,
            "grade_band": grade_band,
            "quiz_text": quiz_text,
        }

    def seed_demo_data(self) -> dict:
        """그래프 확인용 데모 데이터를 생성한다. 이미 기록이 있으면 추가하지 않는다."""
        with connect(self.db_path) as conn:
            existing = conn.execute("SELECT COUNT(*) AS count FROM quiz_attempts").fetchone()["count"]
        if existing:
            return {"created": 0, "message": "이미 학습 기록이 있어 데모 데이터를 추가하지 않았습니다."}

        samples = [
            ("6수01-11", "수학", "5-6", [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]),
            ("4과07-03", "과학", "3-4", [1, 0, 1, 1, 0, 1, 1, 0]),
            ("6국01-06", "국어", "5-6", [1, 1, 1, 0, 1, 1, 0]),
        ]
        created_count = 0
        today = datetime.now()
        for code, subject, grade_band, outcomes in samples:
            for index, outcome in enumerate(outcomes):
                created_at = (today - timedelta(days=len(outcomes) - index)).replace(hour=20, minute=0, second=0)
                self.record_attempt(
                    standard_code=code,
                    subject=subject,
                    grade_band=grade_band,
                    is_correct=bool(outcome),
                    confidence=2 if not outcome else 4,
                    time_spent_sec=90 + index * 5,
                    question_text=f"데모 인출 질문 {code}",
                    user_answer="데모 응답",
                    source_type="demo",
                    created_at=created_at.isoformat(timespec="seconds"),
                )
                created_count += 1
        return {"created": created_count, "message": f"데모 학습 기록 {created_count}개를 추가했습니다."}

    def _standard_trend(self, standard_code: str) -> dict:
        with connect(self.db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                """
                SELECT substr(created_at, 1, 10) AS day,
                       COUNT(*) AS attempts,
                       AVG(is_correct) AS accuracy
                FROM quiz_attempts
                WHERE standard_code = ?
                GROUP BY day
                ORDER BY day
                """,
                (standard_code,),
            )]
        if not rows:
            return {"ma_5": None, "ma_20": None}
        df = pd.DataFrame(rows)
        df["ma_5"] = df["accuracy"].rolling(5, min_periods=1).mean()
        df["ma_20"] = df["accuracy"].rolling(20, min_periods=1).mean()
        last = df.iloc[-1]
        return {"ma_5": round(float(last["ma_5"]), 4), "ma_20": round(float(last["ma_20"]), 4)}

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)

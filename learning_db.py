"""로컬 SQLite 학습 기록 DB.

외부 서비스 없이 VSCode/Streamlit 환경에서 퀴즈 풀이, 복습 일정,
생성 문제 기록을 저장한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path("learning.sqlite3")


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                standard_code TEXT NOT NULL,
                subject TEXT,
                grade_band TEXT,
                source_type TEXT DEFAULT 'manual',
                question_text TEXT,
                user_answer TEXT,
                is_correct INTEGER NOT NULL,
                confidence INTEGER DEFAULT 3,
                time_spent_sec INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_quiz_attempts_created
                ON quiz_attempts(created_at);
            CREATE INDEX IF NOT EXISTS idx_quiz_attempts_standard
                ON quiz_attempts(standard_code);

            CREATE TABLE IF NOT EXISTS review_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER,
                standard_code TEXT NOT NULL,
                subject TEXT,
                grade_band TEXT,
                scheduled_at TEXT NOT NULL,
                interval_days INTEGER NOT NULL,
                status TEXT DEFAULT 'due',
                created_at TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES quiz_attempts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_review_schedule_due
                ON review_schedule(status, scheduled_at);

            CREATE TABLE IF NOT EXISTS generated_quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                standard_code TEXT NOT NULL,
                subject TEXT,
                grade_band TEXT,
                prompt TEXT,
                quiz_text TEXT NOT NULL,
                source TEXT DEFAULT 'template'
            );

            CREATE TABLE IF NOT EXISTS lecture_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                document_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                lecture_date TEXT,
                subject TEXT,
                topics_json TEXT DEFAULT '[]',
                exam_years_json TEXT DEFAULT '[]',
                knowledge_points_json TEXT DEFAULT '[]',
                assignments_json TEXT DEFAULT '[]',
                raw_text TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_lecture_notes_subject
                ON lecture_notes(subject);
            CREATE INDEX IF NOT EXISTS idx_lecture_notes_document
                ON lecture_notes(document_id);
            """
        )

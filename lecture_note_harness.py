#!/usr/bin/env python3
"""인강 요약본 자동 학습자료 편입 하네스.

실제 프로젝트 DB/FAISS를 오염시키지 않도록 임시 디렉터리에서만 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_service import LangChainService
from learning_service import LearningService
from lecture_note_parser import looks_like_lecture_note, parse_lecture_note


SAMPLE = """06-07 강의: 음악 이론 - 당김음, 화음, 노래 형식-Summary
일시: 2026-06-07 16:31:42
요약
이 강의는 현대 동요와 음악 교육에서 중요한 당김음, 화음, 노래 형식, 오스티나토를 다룬다.
지식 포인트
1. 당김음 (Syncopation)
   정의 및 특징: 센박과 여린박의 규칙적인 강세가 뒤바뀌는 리듬.
2. 화음(Chord) 기초 및 장조의 주요 3화음
   으뜸화음, 버금딸림화음, 딸림화음을 구분한다.
8. 기출문제 풀이 분석
   2019년: 라단조 악곡에서 특정 마디에 어울리는 화성단계 딸림화음을 찾는 문제.
질문
- [질문/궁금증 삽입]
과제
- 1. 다장조, 바장조, 사장조의 주요 3화음을 비교하기.
- 2. 리듬 오스티나토와 가락 오스티나토의 차이점을 설명하기.
"""


def main() -> int:
    with TemporaryDirectory() as td:
        tmp = Path(td)
        lecture_file = tmp / "06-07_음악_강의.md"
        lecture_file.write_text(SAMPLE, encoding="utf-8")

        note = parse_lecture_note(SAMPLE, lecture_file.name)
        service = LangChainService(
            openai_api_key="",
            faiss_db_path=str(tmp / "faiss"),
            ocr_cache_path=str(tmp / "ocr"),
            embedding_provider="local",
        )
        result = service.process_text_document(str(lecture_file))
        docs = service._retrieve_documents("당김음 화음 2019년 기출", document_id=result.get("document_id"))
        learning = LearningService(db_path=tmp / "learning.sqlite3")
        saved = learning.save_lecture_note(result["lecture_note"])
        notes = learning.lecture_notes()

    checks = {
        "looks_like_lecture": looks_like_lecture_note(SAMPLE, "06-07_음악_강의.md"),
        "subject_music": note.subject == "음악",
        "topics_extracted": {"당김음", "화음", "오스티나토"}.issubset(set(note.topics)),
        "exam_year_extracted": 2019 in note.exam_years,
        "vector_indexed": result.get("status") == "success" and result.get("document_type") == "lecture_note",
        "retrieval_works": bool(docs) and docs[0].metadata.get("document_type") == "lecture_note",
        "learning_saved": saved.get("indexed") is True and len(notes) == 1,
    }
    output = {
        "passed": all(checks.values()),
        "checks": checks,
        "data": {
            "document_id": result.get("document_id"),
            "chunks_count": result.get("chunks_count"),
            "subject": note.subject,
            "topics": note.topics,
            "exam_years": note.exam_years,
            "lecture_note_id": saved.get("lecture_note_id"),
        },
    }
    Path("lecture_note_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
FastAPI 서버 - LangChain 서비스 래핑
- PDF 업로드 및 처리 API
- 질의응답 API (스트리밍)
- 구글 뉴스 검색 API
"""

# ============================================
# 1. 필요한 라이브러리 불러오기
# ============================================
import os
import shutil
import hashlib
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException  # FastAPI 관련
from fastapi.responses import StreamingResponse               # 스트리밍 응답용
from fastapi.middleware.cors import CORSMiddleware             # CORS 설정용
from pydantic import BaseModel                                 # 데이터 모델 정의용
from dotenv import load_dotenv

from langchain_service import LangChainService  # LangChain 서비스 불러오기
from comparison_service import CurriculumComparisonService
from agents.supervisor_agent import SupervisorAgent
from learning_service import LearningService
from lecture_note_parser import looks_like_lecture_note, parse_lecture_note


APP_VERSION = "2026-06-23-document-aware-upload"


# ============================================
# 2. FastAPI 앱 생성 및 설정
# ============================================
app = FastAPI(title="PDF QA & News Search Service")

# CORS 설정 (다른 도메인에서의 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],      # 모든 HTTP 메서드 허용
    allow_headers=["*"],      # 모든 헤더 허용
)


# ============================================
# 3. LangChain 서비스 초기화
# ============================================
# 환경변수에서 API 키 가져오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LangChain 서비스 인스턴스 생성
langchain_service = LangChainService(
    openai_api_key=OPENAI_API_KEY,
    faiss_db_path=os.getenv("VECTOR_DB_PATH", "faiss_curriculum"),
    ocr_cache_path="ocr_cache",
    ocr_backend=os.getenv("OCR_BACKEND", "auto"),
    embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
)
comparison_service = CurriculumComparisonService(
    mapping_path="curriculum_mapping/mappings.json",
    vector_db_path=os.getenv("VECTOR_DB_PATH", "faiss_curriculum"),
)
supervisor_agent = SupervisorAgent(
    comparison_service=comparison_service,
    langchain_service=langchain_service,
)
learning_service = LearningService()

# 업로드 파일 저장 폴더 생성
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)
CURRICULUM_DIR = Path("parsed_exams_native") / "교육과정"


# ============================================
# 4. 요청/응답 데이터 모델 정의
# ============================================
class QueryRequest(BaseModel):
    """질문 요청 모델"""
    question: str  # 사용자 질문
    document_id: str | None = None  # 방금 업로드한 문서 등 특정 문서 우선 검색


class QueryResponse(BaseModel):
    """질문 응답 모델"""
    answer: str      # AI 답변
    documents: list  # 관련 문서 목록


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str  # 검색어


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    status: str  # 성공/실패 상태
    answer: str  # 검색 결과
    query: str   # 검색어


class LibraryResponse(BaseModel):
    status: str
    files_count: int
    indexed_count: int
    skipped_count: int
    failed_count: int
    results: list


class ComparisonRequest(BaseModel):
    subject: str | None = None
    grade_band: str | None = None
    domain: str | None = None
    query: str = ""
    limit: int = 10


class LessonCoachRequest(BaseModel):
    query: str
    subject: str | None = None
    grade: int | None = None
    limit: int = 6


class LearningAttemptRequest(BaseModel):
    standard_code: str
    subject: str | None = None
    grade_band: str | None = None
    is_correct: bool
    confidence: int = 3
    time_spent_sec: int = 0
    question_text: str = ""
    user_answer: str = ""
    source_type: str = "manual"
    created_at: str | None = None


class VariantQuizRequest(BaseModel):
    standard_code: str
    subject: str | None = None
    grade_band: str | None = None
    weakness_note: str = ""


class ReviewCompleteRequest(BaseModel):
    standard_code: str
    completed_at: str | None = None


# ============================================
# 5. API 엔드포인트 정의
# ============================================

# ---------------------------
# 루트 엔드포인트
# ---------------------------
@app.get("/")
def read_root():
    """
    루트 엔드포인트 - API 정보 반환
    """
    return {
        "message": "PDF QA & News Search Service API",
        "version": APP_VERSION,
        "endpoints": {
            "upload": "/upload-pdf",
            "upload_document": "/upload-document",
            "ingest_library": "/ingest-library",
            "query_stream": "/query/stream",
            "documents": "/documents",
            "comparison": "/comparison",
            "lesson_coach": "/lesson-coach",
            "learning_metrics": "/learning/metrics",
            "learning_attempt": "/learning/attempt",
            "search": "/search",
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "features": {
            "document_aware_query": True,
            "upload_document": True,
            "lecture_note_ingestion": True,
        },
    }


# ---------------------------
# PDF 업로드 엔드포인트
# ---------------------------
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    """
    PDF 파일 업로드 및 처리
    
    Args:
        file: 업로드된 PDF 파일
        
    Returns:
        처리 결과 (청크 수, 페이지 수 등)
    """
    # 파일 확장자 확인
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")
    
    try:
        result = _save_and_process_upload(file)
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        return _upload_response(result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 실패: {str(e)}")


@app.post("/upload-document")
def upload_document(file: UploadFile = File(...)):
    """PDF, txt, md 자료를 업로드하고 자동으로 학습데이터에 추가한다."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="PDF, TXT, MD 파일만 업로드 가능합니다")
    try:
        result = _save_and_process_upload(file)
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
        return _upload_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 처리 실패: {str(e)}")


def _save_and_process_upload(file: UploadFile) -> dict:
    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        result = langchain_service.process_pdf(str(file_path))
    else:
        result = langchain_service.process_text_document(str(file_path))
    result["filename"] = safe_filename
    return result


def _upload_response(result: dict) -> dict:
    learning_data = None
    if result.get("document_type") == "lecture_note" and result.get("lecture_note"):
        learning_data = learning_service.save_lecture_note(result["lecture_note"])
    return {
        "message": result["message"],
        "filename": result["filename"],
        "chunks_count": result["chunks_count"],
        "pages_count": result["pages_count"],
        "document_id": result["document_id"],
        "extraction_methods": result["extraction_methods"],
        "warnings": result["warnings"],
        "indexed": result.get("indexed", True),
        "document_type": result.get("document_type"),
        "lecture_note": result.get("lecture_note"),
        "learning_data": learning_data,
    }


@app.get("/uploaded-documents")
def uploaded_documents():
    """업로드된 PDF/TXT/MD 목록과 document_id를 반환한다."""
    documents = []
    files = [
        path for path in UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
    ]
    for pdf in sorted(files, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            if pdf.suffix.lower() == ".pdf":
                extracted = langchain_service.ocr_service.extract_pdf(str(pdf))
                document_id = extracted.document_id
                pages_count = len(extracted.pages)
                document_type = langchain_service._document_metadata(str(pdf), document_id).get("document_type")
            else:
                text = pdf.read_text(encoding="utf-8")
                if looks_like_lecture_note(text, pdf.name):
                    note = parse_lecture_note(text, pdf.name)
                    document_id = note.document_id
                    document_type = "lecture_note"
                else:
                    document_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    document_type = "text_reference"
                pages_count = 1
            documents.append({
                "filename": pdf.name,
                "document_id": document_id,
                "pages_count": pages_count,
                "document_type": document_type,
                "modified_at": pdf.stat().st_mtime,
            })
        except Exception as exc:
            documents.append({
                "filename": pdf.name,
                "document_id": None,
                "error": str(exc),
                "modified_at": pdf.stat().st_mtime,
            })
    return {"documents": documents}


@app.get("/learning/lecture-notes")
def lecture_notes(limit: int = 20, subject: str | None = None):
    """저장된 인강 요약본 학습자료 목록을 반환한다."""
    return {"lecture_notes": learning_service.lecture_notes(limit=max(1, min(limit, 100)), subject=subject)}


@app.post("/ingest-library", response_model=LibraryResponse)
def ingest_library():
    """프로젝트의 교육과정·평가기준 PDF를 중복 없이 일괄 색인한다."""
    if not CURRICULUM_DIR.exists():
        raise HTTPException(status_code=404, detail="교육과정 자료 폴더를 찾지 못했습니다")
    pdfs = sorted(CURRICULUM_DIR.glob("*.pdf"))
    results = []
    for pdf in pdfs:
        result = langchain_service.process_pdf(str(pdf))
        results.append({"filename": pdf.name, **result})
    indexed = sum(r.get("indexed") is True for r in results)
    skipped = sum(r.get("status") == "success" and r.get("indexed") is False for r in results)
    failed = sum(r.get("status") != "success" for r in results)
    return {
        "status": "success" if failed == 0 else "partial_success",
        "files_count": len(pdfs),
        "indexed_count": indexed,
        "skipped_count": skipped,
        "failed_count": failed,
        "results": results,
    }


@app.get("/comparison/filters")
def comparison_filters():
    comparison_service.refresh()
    return comparison_service.filters()


@app.post("/comparison")
def compare_curricula(request: ComparisonRequest):
    comparison_service.refresh()
    result = comparison_service.compare(
        subject=request.subject,
        grade_band=request.grade_band,
        domain=request.domain,
        query=request.query,
        limit=max(1, min(request.limit, 30)),
    )
    result["answer"] = langchain_service.generate_comparison_answer(request.query, result)
    return result


@app.post("/lesson-coach")
def lesson_coach(request: LessonCoachRequest):
    """총론·평가기준 총괄 에이전트가 교과·학년 관점으로 수업-임용 답변을 생성한다."""
    return supervisor_agent.run(
        query=request.query,
        subject=request.subject,
        grade=request.grade,
        limit=max(1, min(request.limit, 12)),
    )


@app.post("/learning/attempt")
def record_learning_attempt(request: LearningAttemptRequest):
    """퀴즈/인출 결과를 저장하고 에빙하우스 복습 일정을 생성한다."""
    return learning_service.record_attempt(**request.model_dump())


@app.get("/learning/metrics")
def learning_metrics(days: int = 60, subject: str | None = None):
    """일간 정답률과 5일/20일 이동평균선을 반환한다."""
    return {"metrics": learning_service.daily_metrics(days=max(1, min(days, 365)), subject=subject)}


@app.get("/learning/metrics/aggregate")
def learning_aggregate_metrics(days: int = 180, period: str = "W", subject: str | None = None):
    """주간/월간 정답률과 5구간/20구간 이동평균선을 반환한다."""
    return {
        "metrics": learning_service.aggregate_metrics(
            days=max(1, min(days, 730)),
            period=period,
            subject=subject,
        )
    }


@app.get("/learning/weak-standards")
def weak_standards(limit: int = 10):
    """정답률·이동평균 하향 돌파 기반 취약 성취기준을 반환한다."""
    return {"weak_standards": learning_service.weak_standards(limit=max(1, min(limit, 50)))}


@app.get("/learning/review-due")
def review_due(limit: int = 20):
    """오늘까지 복습 예정인 성취기준을 반환한다."""
    return {"due_reviews": learning_service.due_reviews(limit=max(1, min(limit, 100)))}


@app.post("/learning/review-complete")
def complete_review(request: ReviewCompleteRequest):
    """오늘까지 도래한 특정 성취기준 복습 일정을 완료 처리한다."""
    return learning_service.complete_reviews(**request.model_dump())


@app.post("/learning/generate-variant")
def generate_variant_quiz(request: VariantQuizRequest):
    """취약 성취기준 기반 로컬 템플릿 변형 문제를 생성한다."""
    return learning_service.generate_variant(**request.model_dump())


@app.post("/learning/seed-demo")
def seed_learning_demo():
    """그래프 확인용 데모 학습 기록을 생성한다."""
    return learning_service.seed_demo_data()


# ---------------------------
# 질의응답 스트리밍 엔드포인트
# ---------------------------
@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """
    질문에 대한 답변 생성 (스트리밍)
    
    Args:
        request: 질문 요청 (question 필드 포함)
        
    Returns:
        스트리밍 응답
    """
    
    #########################################################
    # 문제2-(2):
    # fastapi로 스트리밍 답변 api 구성
    # 에러 헨들링
    
    def generate():
        for chunk in langchain_service.query_stream(
            request.question,
            document_id=request.document_id,
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")
    
    #########################################################


# ---------------------------
# 관련 문서 검색 엔드포인트
# ---------------------------
@app.post("/documents")
def get_documents(request: QueryRequest):
    """
    질문과 관련된 문서 검색
    
    Args:
        request: 질문 요청
        
    Returns:
        관련 문서 리스트 (상위 5개)
    """
    documents = langchain_service.get_retrieved_documents(
        request.question,
        document_id=request.document_id,
    )
    return {"documents": documents}


# ---------------------------
# 뉴스 검색 엔드포인트
# ---------------------------
@app.post("/search", response_model=SearchResponse)
def search_news(request: SearchRequest):
    """
    구글 뉴스 검색
    
    Args:
        request: 검색 요청 (query 필드 포함)
        
    Returns:
        검색 결과
    """
    result = langchain_service.search_google_news(request.query)
    return result


# ============================================
# 6. 서버 실행
# ============================================
if __name__ == "__main__":
    import uvicorn
    # Streamlit 앱(app.py)은 http://localhost:8010 으로 이 API를 호출한다.
    uvicorn.run(app, host="127.0.0.1", port=8010)

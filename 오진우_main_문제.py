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
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException  # FastAPI 관련
from fastapi.responses import StreamingResponse               # 스트리밍 응답용
from fastapi.middleware.cors import CORSMiddleware             # CORS 설정용
from pydantic import BaseModel                                 # 데이터 모델 정의용

from langchain_service import LangChainService  # LangChain 서비스 불러오기


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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요")

# LangChain 서비스 인스턴스 생성
langchain_service = LangChainService(
    openai_api_key=OPENAI_API_KEY,
    faiss_db_path="faiss_db"
)

# 업로드 파일 저장 폴더 생성
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================
# 4. 요청/응답 데이터 모델 정의
# ============================================
class QueryRequest(BaseModel):
    """질문 요청 모델"""
    question: str  # 사용자 질문


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
        "endpoints": {
            "upload": "/upload-pdf",
            "query_stream": "/query/stream",
            "documents": "/documents",
            "search": "/search",
        }
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
        # 파일 저장
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # PDF 처리 (LangChain 서비스 호출)
        result = langchain_service.process_pdf(str(file_path))
        
        if result["status"] == "success":
            return {
                "message": result["message"],
                "filename": file.filename,
                "chunks_count": result["chunks_count"],
                "pages_count": result["pages_count"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 실패: {str(e)}")


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
try:
    def generate():
        for chunk in langchain_service.query_stream(request.question):
            yield chunk
        
        return StreamingResponse(
           generate(),
           media_type="text/plain"
    )
    
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"failure to make the answer for streaming: {str(e)}"
    )
    
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
    documents = langchain_service.get_retrieved_documents(request.question)
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
    # 서버 실행: localhost:8010
    uvicorn.run(app, host="0.0.0.0", port=8010)

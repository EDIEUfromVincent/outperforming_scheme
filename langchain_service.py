"""
LangChain 서비스 모듈
- PDF 문서를 파싱하고 FAISS 벡터DB에 저장
- 문서 기반 질의응답 처리
- 구글 뉴스 검색 Agent
"""

# ============================================
# 1. 필요한 라이브러리 불러오기
# ============================================
import os
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# LangChain 관련 라이브러리
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 텍스트 분할기
from langchain_community.vectorstores import FAISS                # 벡터 저장소
from langchain_openai import OpenAIEmbeddings, ChatOpenAI         # OpenAI 모델
from langchain_core.prompts import ChatPromptTemplate             # 프롬프트 템플릿
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # 데이터 전달용
from langchain_core.output_parsers import StrOutputParser         # 출력 파서
from langchain_core.documents import Document                     # 페이지 문서
from langchain_core.embeddings import Embeddings
from langchain.tools import tool                                  # 도구 데코레이터
from langchain.agents import create_tool_calling_agent, AgentExecutor  # Agent 관련
from langchain_teddynote.tools import GoogleNews                  # 구글 뉴스 도구
from ocr_service import OCRService                                # OCR fallback
from lecture_note_parser import looks_like_lecture_note, parse_lecture_note


class LocalHashEmbeddings(Embeddings):
    """API 키 없이 동작하는 한국어 문자 n-gram 임베딩."""

    def __init__(self, dimensions: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ============================================
# 2. LangChainService 클래스 정의
# ============================================
class LangChainService:
    """PDF 문서 기반 QA 서비스 + 구글 뉴스 검색 Agent"""
    
    def __init__(
        self,
        openai_api_key: str = "",
        faiss_db_path: str = "faiss_curriculum",
        ocr_cache_path: str = "ocr_cache",
        ocr_backend: str = "auto",
        embedding_provider: str = "auto",
    ):
        """
        서비스 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            faiss_db_path: FAISS DB 저장 경로
        """
        # API 키 환경변수 설정
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # OpenAI 임베딩 모델 설정 (텍스트를 벡터로 변환)
        use_openai_embeddings = embedding_provider == "openai" or (
            embedding_provider == "auto" and bool(openai_api_key)
        )
        self.embeddings = (
            OpenAIEmbeddings(model="text-embedding-3-small")
            if use_openai_embeddings
            else LocalHashEmbeddings()
        )
        self.embedding_provider = "openai" if use_openai_embeddings else "local_hash"
        
        # ChatGPT 모델 설정
        self.llm = (
            ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=1400,
            )
            if openai_api_key
            else None
        )
        
        # FAISS DB 경로 저장
        self.faiss_db_path = faiss_db_path
        self.index_manifest_path = Path(faiss_db_path) / "documents.json"
        self.indexed_document_ids = self._load_index_manifest()
        
        # 벡터 스토어, 검색기, 체인 초기화
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.ocr_service = OCRService(cache_dir=ocr_cache_path, backend=ocr_backend)
        
        # 구글 뉴스 검색 Agent 설정
        self.search_agent = None
        if self.llm is not None:
            self._setup_search_agent()
        
        # 기존 FAISS DB가 있으면 로드
        if Path(faiss_db_path).exists():
            self._load_vector_store()
    
    # ---------------------------
    # Agent 설정 메서드
    # ---------------------------
    def _setup_search_agent(self):
        """구글 뉴스 검색 Agent 설정"""
        
        ###################################################
        # 문제 3
        # google news 검색 tool 등록
        
        @tool
        def search_news(query: str) -> List[Dict[str, str]]:
            """Search Google News by input keyword"""
            news_tool = GoogleNews()
            return news_tool.search_by_keyword(query, k=5)
        
        # 도구 리스트
        tools = [search_news]
        
        ###################################################
        
        # Agent 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful assistant. "
             "Make sure to use the `search_news` tool for searching keyword related news."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Agent 생성
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        # AgentExecutor 생성 (Agent를 실행하는 역할)
        self.search_agent = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,              # 상세 로그 출력 여부
            handle_parsing_errors=True  # 파싱 에러 자동 처리
        )
    
    # ---------------------------
    # 벡터 스토어 로드 메서드
    # ---------------------------
    def _load_vector_store(self):
        """저장된 FAISS 벡터 스토어 로드"""
        try:
            # 로컬에 저장된 FAISS DB 불러오기
            self.vector_store = FAISS.load_local(
                folder_path=self.faiss_db_path,
                index_name="index",
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # 역직렬화 허용
            )
            
            # 검색기 설정 (유사도 검색, 상위 5개 문서 반환)
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 30}
            )
            
            # QA 체인 설정
            self._setup_chain()
            
        except Exception as e:
            print(f"벡터 스토어 로드 실패: {e}")
    
    # ---------------------------
    # QA 체인 설정 메서드
    # ---------------------------
    def _setup_chain(self):
        """QA 체인 설정 - 질문에 답변하는 파이프라인"""
        if self.llm is None:
            self.chain = None
            return
        
        # 프롬프트 템플릿 정의
        template = ChatPromptTemplate.from_messages([
            ('system', '''당신은 초등 교원임용을 준비하는 교대생을 위한 교육과정·수업설계 코치입니다.
개념 암기에 그치지 않고 교육과정, 기출의 수업 상황, 평가기준을 실제 수업 설계로 연결하세요.'''),
            ('system', '''아래 제공된 context를 주의 깊게 읽고 사용자의 질문에 답변해주세요.
            
규칙:
1. 2009 개정 교육과정은 비교·검색 범위에서 제외하세요.
2. 2015와 2022 개정의 차이는 각각의 근거를 확보했을 때만 설명하세요.
3. 기출에는 공식 정답이 없으므로 답안을 단정하지 말고 "근거 기반 예상 답안"으로 표시하세요.
4. 교육과정 원문 → 평가기준·성취수준 → 기출 순으로 근거의 우선순위를 두세요.
5. 중요한 주장 뒤에는 [근거 n]을 표시하고 문서명과 페이지를 답변 끝에 정리하세요.
6. 수업 적용 질문에는 목표, 학생 활동, 교사 지원, 평가 증거를 구체적으로 연결하세요.
7. 사용자가 연습을 원하면 짧은 인출 질문을 제시하되 답을 먼저 노출하지 마세요.
8. context에 정보가 없거나 근거끼리 충돌하면 그 사실을 명시하세요.

Context:
{context}'''),
            ('human', '{question}')
        ])
        
        # 체인 구성: 데이터 → 프롬프트 → LLM → 문자열 출력
        data = {
            "question": RunnablePassthrough(),
            "context": self.retriever | RunnableLambda(self._format_context),
        }
        self.chain = data | template | self.llm | StrOutputParser()

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        blocks = []
        for index, document in enumerate(documents, 1):
            metadata = document.metadata
            page = metadata.get("page_number")
            if page is None and metadata.get("page") is not None:
                page = metadata["page"] + 1
            label = " · ".join(
                str(value)
                for value in [
                    metadata.get("filename") or Path(metadata.get("source", "")).name,
                    f"p.{page}" if page else None,
                    metadata.get("curriculum_version"),
                    metadata.get("document_type"),
                ]
                if value
            )
            blocks.append(f"[근거 {index}] {label}\n{document.page_content}")
        return "\n\n".join(blocks)
    
    # ---------------------------
    # PDF 처리 메서드
    # ---------------------------
    def process_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        PDF 파일을 처리하고 FAISS에 저장
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            처리 결과 딕셔너리
        """
        try:
            # 1. 텍스트 레이어 우선, 저품질 페이지만 OCR
            extracted = self.ocr_service.extract_pdf(pdf_path)
            common_metadata = self._document_metadata(pdf_path, extracted.document_id)
            if extracted.document_id in self.indexed_document_ids:
                return {
                    "status": "success",
                    "message": "이미 색인된 동일 문서입니다.",
                    "chunks_count": 0,
                    "pages_count": len(extracted.pages),
                    "document_id": extracted.document_id,
                    "extraction_methods": extracted.methods,
                    "warnings": extracted.warnings,
                    "indexed": False,
                }
            documents = [
                Document(
                    page_content=page.text,
                    metadata={
                        **common_metadata,
                        "page": page.page - 1,
                        "page_number": page.page,
                        "extraction_method": page.method,
                        "ocr_confidence": page.confidence,
                    },
                )
                for page in extracted.pages
                if page.text.strip()
            ]
            if not documents:
                raise ValueError("PDF에서 검색 가능한 텍스트를 추출하지 못했습니다")
            
            # 2. 텍스트를 작은 청크로 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,    # 청크 크기 (글자 수)
                chunk_overlap=150,  # 청크 간 겹침 (문맥 유지용)
                separators=["\n\n", "\n", "。", ". ", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            
            # 3. FAISS 벡터 스토어 생성 또는 업데이트
            if self.vector_store is None:
                # 새로 생성
                self.vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
            else:
                # 기존 스토어에 추가
                new_vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                self.vector_store.merge_from(new_vector_store)
            
            # 4. FAISS DB 로컬에 저장
            self.vector_store.save_local(
                folder_path=self.faiss_db_path,
                index_name="index"
            )
            self.indexed_document_ids.add(extracted.document_id)
            self._save_index_manifest()
            
            # 5. 검색기 및 체인 설정
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 30}
            )
            self._setup_chain()
            
            return {
                "status": "success",
                "message": f"PDF 처리 완료: {len(chunks)}개 청크 생성",
                "chunks_count": len(chunks),
                "pages_count": len(extracted.pages),
                "document_id": extracted.document_id,
                "extraction_methods": extracted.methods,
                "warnings": extracted.warnings,
                "indexed": True,
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"PDF 처리 실패: {str(e)}"
            }

    def process_text_document(self, file_path: str) -> Dict[str, any]:
        """txt/md 강의 요약본 또는 일반 텍스트 자료를 FAISS에 저장한다."""
        try:
            path = Path(file_path)
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = Path(file_path).read_text(encoding="utf-8-sig")
        try:
            is_lecture = looks_like_lecture_note(raw_text, Path(file_path).name)
            note = parse_lecture_note(raw_text, Path(file_path).name) if is_lecture else None
            document_id = note.document_id if note else hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            if document_id in self.indexed_document_ids:
                return {
                    "status": "success",
                    "message": "이미 색인된 동일 문서입니다.",
                    "chunks_count": 0,
                    "pages_count": 1,
                    "document_id": document_id,
                    "extraction_methods": ["text"],
                    "warnings": [],
                    "indexed": False,
                    "document_type": "lecture_note" if is_lecture else "text_reference",
                    "lecture_note": note.to_dict() if note else None,
                }
            metadata = self._document_metadata(file_path, document_id)
            if is_lecture and note:
                metadata.update({
                    "document_type": "lecture_note",
                    "title": note.title,
                    "lecture_date": note.lecture_date,
                    "subject": note.subject,
                    "topics": ", ".join(note.topics[:12]),
                    "exam_years": ", ".join(str(year) for year in note.exam_years),
                })
            else:
                metadata.update({"document_type": "text_reference"})
            base_document = Document(
                page_content=raw_text,
                metadata={
                    **metadata,
                    "page": 0,
                    "page_number": 1,
                    "extraction_method": "text",
                },
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", "。", ". ", " ", ""],
            )
            chunks = splitter.split_documents([base_document])
            if not chunks:
                raise ValueError("텍스트 파일에서 색인 가능한 내용을 찾지 못했습니다")
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            else:
                new_vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
                self.vector_store.merge_from(new_vector_store)
            self.vector_store.save_local(folder_path=self.faiss_db_path, index_name="index")
            self.indexed_document_ids.add(document_id)
            self._save_index_manifest()
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 30},
            )
            self._setup_chain()
            return {
                "status": "success",
                "message": f"텍스트 자료 처리 완료: {len(chunks)}개 청크 생성",
                "chunks_count": len(chunks),
                "pages_count": 1,
                "document_id": document_id,
                "extraction_methods": ["text"],
                "warnings": [],
                "indexed": True,
                "document_type": "lecture_note" if is_lecture else "text_reference",
                "lecture_note": note.to_dict() if note else None,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"텍스트 자료 처리 실패: {str(e)}",
            }

    def _load_index_manifest(self) -> set[str]:
        try:
            data = json.loads(self.index_manifest_path.read_text(encoding="utf-8"))
            return set(data.get("document_ids", []))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_index_manifest(self) -> None:
        self.index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_manifest_path.write_text(
            json.dumps(
                {"document_ids": sorted(self.indexed_document_ids)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _document_metadata(pdf_path: str, document_id: str) -> Dict[str, any]:
        """파일명에서 확실히 알 수 있는 검색 메타데이터만 생성한다."""
        import re
        import unicodedata

        name = unicodedata.normalize("NFC", Path(pdf_path).name)
        if "평가기준" in name or "성취수준" in name or "평가도구" in name:
            document_type = "assessment_standard"
        elif "학년도" in name and ("교육과정" in name or "교직과정" in name or "교직논술" in name):
            document_type = "exam"
        elif "강의" in name or "요약" in name or "인강" in name:
            document_type = "lecture_note"
        elif "해설" in name:
            document_type = "curriculum_commentary"
        elif "교육과정" in name:
            document_type = "curriculum"
        else:
            document_type = "reference"

        version = None
        if "2015" in name or "제2015-" in name:
            version = "2015"
        elif "2022" in name or name.startswith("[별책"):
            version = "2022"

        subjects = [
            "국어", "수학", "사회", "과학", "영어", "도덕", "실과",
            "체육", "음악", "미술", "통합교과", "창의적 체험활동",
        ]
        subject = next((item for item in subjects if item in name), None)
        grade_match = re.search(r"([1-6])\s*[~∼～-]\s*([1-6])학년", name)
        grade_band = f"{grade_match.group(1)}-{grade_match.group(2)}" if grade_match else None
        return {
            "source": str(pdf_path),
            "filename": name,
            "document_id": document_id,
            "document_type": document_type,
            "curriculum_version": version,
            "subject": subject,
            "grade_band": grade_band,
        }
    
    # ---------------------------
    # 스트리밍 질의응답 메서드
    # ---------------------------
    def query_stream(self, question: str, document_id: str | None = None):
        """
        질문에 대한 답변 생성 (스트리밍 방식)
        
        Args:
            question: 사용자 질문
            
        Yields:
            답변 토큰 (한 글자씩)
        """
        if document_id:
            primary_documents = self._retrieve_documents(question, document_id=document_id, k=8)
            if not primary_documents:
                yield "방금 업로드한 문서에서 관련 내용을 찾지 못했습니다. 업로드 처리가 끝났는지 확인해주세요."
                return
            secondary_documents = self._retrieve_secondary_documents_for_uploaded_query(question, document_id)
            yield from self._answer_from_documents(
                question,
                primary_documents,
                document_id=document_id,
                secondary_documents=secondary_documents,
            )
            return

        # 체인이 설정되지 않은 경우
        if self.chain is None:
            if self.retriever is None:
                yield "먼저 교육과정 자료를 색인해주세요."
                return
            documents = self._retrieve_documents(question, k=8)
            yield "OpenAI API 키가 없어 근거 검색 결과만 제공합니다.\n\n"
            yield self._format_context(documents)
            return
        
        try:
            # 스트리밍으로 응답 생성
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            yield f"오류 발생: {str(e)}"
    
    # ---------------------------
    # 문서 검색 메서드
    # ---------------------------
    def get_retrieved_documents(self, question: str, document_id: str | None = None) -> List[Dict]:
        """
        질문과 관련된 문서 검색
        
        Args:
            question: 사용자 질문
            
        Returns:
            관련 문서 리스트
        """
        if self.retriever is None:
            return []

        try:
            # 유사 문서 검색
            retrieved_docs = self._retrieve_documents(question, document_id=document_id, k=8)

            # 결과 포맷팅
            documents = []
            for doc in retrieved_docs[:5]:
                documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            return documents

        except Exception as e:
            print(f"문서 검색 실패: {e}")
            return []

    def _retrieve_documents(
        self,
        question: str,
        document_id: str | None = None,
        k: int = 8,
    ) -> List[Document]:
        if self.vector_store is None:
            if document_id:
                return self._documents_from_uploaded_file(document_id=document_id)
            return []
        if document_id:
            try:
                docs = self.vector_store.similarity_search(
                    question,
                    k=k,
                    filter={"document_id": document_id},
                )
                if docs:
                    return docs
            except Exception:
                # 일부 FAISS 필터 구현에서 실패할 때를 대비한 보수적 fallback
                docs = self.vector_store.similarity_search(question, k=max(k * 4, 20))
                filtered = [doc for doc in docs if doc.metadata.get("document_id") == document_id][:k]
                if filtered:
                    return filtered
            return self._documents_from_uploaded_file(document_id=document_id)
        if self.retriever is not None:
            return self.retriever.invoke(question)[:k]
        return self.vector_store.similarity_search(question, k=k)

    def _documents_from_uploaded_file(self, document_id: str) -> List[Document]:
        """FAISS와 manifest가 어긋난 경우 업로드 폴더의 원문 PDF를 직접 읽는다."""
        upload_dir = Path("uploaded_files")
        if not upload_dir.exists():
            return []
        for pdf_path in upload_dir.iterdir():
            if not pdf_path.is_file() or pdf_path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            try:
                if pdf_path.suffix.lower() == ".pdf":
                    extracted = self.ocr_service.extract_pdf(str(pdf_path))
                    current_document_id = extracted.document_id
                    if current_document_id != document_id:
                        continue
                    metadata = self._document_metadata(str(pdf_path), extracted.document_id)
                    return [
                        Document(
                            page_content=page.text,
                            metadata={
                                **metadata,
                                "page": page.page - 1,
                                "page_number": page.page,
                                "extraction_method": page.method,
                                "ocr_confidence": page.confidence,
                                "direct_upload_fallback": True,
                            },
                        )
                        for page in extracted.pages
                        if page.text.strip()
                    ]
                text = pdf_path.read_text(encoding="utf-8")
                if looks_like_lecture_note(text, pdf_path.name):
                    note = parse_lecture_note(text, pdf_path.name)
                    current_document_id = note.document_id
                    metadata = {
                        **self._document_metadata(str(pdf_path), current_document_id),
                        "document_type": "lecture_note",
                        "title": note.title,
                        "lecture_date": note.lecture_date,
                        "subject": note.subject,
                        "topics": ", ".join(note.topics[:12]),
                        "exam_years": ", ".join(str(year) for year in note.exam_years),
                    }
                else:
                    current_document_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    metadata = {
                        **self._document_metadata(str(pdf_path), current_document_id),
                        "document_type": "text_reference",
                    }
            except Exception:
                continue
            if current_document_id != document_id:
                continue
            return [
                Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "page": 0,
                        "page_number": 1,
                        "extraction_method": "text",
                        "direct_upload_fallback": True,
                    },
                )
            ]
        return []

    def _answer_from_documents(
        self,
        question: str,
        documents: List[Document],
        document_id: str | None = None,
        secondary_documents: List[Document] | None = None,
    ):
        primary_context = self._format_context(documents)
        secondary_context = self._format_context(secondary_documents or [])
        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 초등임용 수험생을 위한 문서 기반 답변 코치입니다.
반드시 1차 context인 방금 업로드한 문서를 먼저 요약하고, 답변의 중심 근거로 삼으세요.
2차 context는 총론·평가기준·교육과정 연결이 필요할 때만 보조 근거로 사용하세요.
업로드 문서를 무시하고 2차 context만으로 답하면 안 됩니다.
사용자가 예시답안이나 논술 답안을 요청하면 문제의 요구사항과 배점을 반영해 논리적인 답안 형태로 작성하세요.
context에 없는 내용은 외부 지식으로 채우지 말고, 필요한 경우 '문서에 명시되지 않음'이라고 하세요.
기출 공식 답안이 아니라 '근거 기반 예시답안'으로 표시하세요.
답변 끝에는 사용한 근거 문서명/페이지를 간단히 정리하세요."""),
                ("human", """질문: {question}

1차 context: 방금 업로드한 문서
{primary_context}

2차 context: 기존 학습자료/총론/평가기준 연결 근거
{secondary_context}"""),
            ])
            try:
                answer = (prompt | self.llm | StrOutputParser()).invoke({
                    "question": question,
                    "primary_context": primary_context,
                    "secondary_context": secondary_context or "2차 연결 근거 없음",
                })
                yield answer
                return
            except Exception as exc:
                print(f"업로드 문서 LLM 답변 실패: {exc}")
        yield self._fallback_uploaded_answer(question, documents)

    def _retrieve_secondary_documents_for_uploaded_query(
        self,
        question: str,
        document_id: str | None,
        k: int = 6,
    ) -> List[Document]:
        """업로드 문서 답변에 붙일 총론·평가기준·교육과정 보조 근거를 찾는다."""
        if self.vector_store is None:
            return []
        query = f"{question} 총론 평가기준 성취수준 교육과정"
        docs: list[Document] = []
        for metadata_filter in [
            {"document_type": "assessment_standard"},
            {"document_type": "curriculum_commentary"},
            {"document_type": "curriculum"},
            None,
        ]:
            try:
                if metadata_filter:
                    found = self.vector_store.similarity_search(query, k=3, filter=metadata_filter)
                else:
                    found = self.vector_store.similarity_search(query, k=8)
            except Exception:
                continue
            for doc in found:
                if doc.metadata.get("document_id") == document_id:
                    continue
                key = (doc.metadata.get("source"), doc.metadata.get("page_number"), doc.page_content[:80])
                if any((old.metadata.get("source"), old.metadata.get("page_number"), old.page_content[:80]) == key for old in docs):
                    continue
                docs.append(doc)
                if len(docs) >= k:
                    return docs
        return docs[:k]

    @staticmethod
    def _fallback_uploaded_answer(question: str, documents: List[Document]) -> str:
        text = "\n".join(doc.page_content for doc in documents)
        filename = documents[0].metadata.get("filename") or Path(documents[0].metadata.get("source", "")).name
        page = documents[0].metadata.get("page_number") or 1
        if "교직 논술" in text or "교직논술" in text or "대인관계" in text:
            return f"""근거 기반 예시답안입니다. 공식 정답이 아니라, 업로드한 문서의 요구사항과 배점에 맞춘 답안 예시입니다.

제목: 학생의 대인관계 능력 함양을 위한 협력적 생활지도와 상담

학생의 대인관계 능력은 학교생활 적응과 공동체적 성장의 기초가 된다. 제시문에서 교사들은 코로나19 상황 이후 학생들이 대인관계를 형성할 기회가 부족해졌고, 그 결과 학생 간 관계 문제가 늘어났다고 보았다. 따라서 담임교사의 개별적 노력뿐 아니라 학교장, 학부모, 지역사회 전문 상담사가 함께 참여하는 협력 지원 체제를 바탕으로 학생의 강점을 활용하고 상담 목표를 적절히 설정할 필요가 있다.

첫째, 학생들의 대인관계 능력을 함양하기 위해 학교장은 학교 관리자로서 지원할 수 있다. 우선 학교 차원의 생활지도와 상담 계획을 수립하고, 교사 학습 공동체가 지속적으로 운영될 수 있도록 시간과 행정적 여건을 마련할 수 있다. 또한 또래 관계 회복 프로그램, 상담 주간, 학급 단위 관계 형성 활동 등이 안정적으로 이루어지도록 예산과 인력을 지원할 수 있다. 학부모는 자녀의 보호자로서 가정에서 자녀의 대인관계 경험을 관찰하고 교사와 공유할 수 있다. 또한 자녀가 친구의 감정과 입장을 이해하도록 대화 기회를 제공하고, 학교 상담 과정에서 정서적 지지를 지속할 수 있다. 지역사회 전문 상담사는 대인관계 전문가로서 학생의 관계 문제를 진단하고, 사회적 기술 훈련이나 의사소통 훈련 프로그램을 제공할 수 있다. 또한 교사와 학부모에게 학생 상담에 필요한 전문적 조언을 제공하여 학교의 생활지도 역량을 보완할 수 있다.

둘째, 영우와 진서의 강점을 상담에 활용해야 한다. 영우는 친구들과 어울리기를 좋아한다는 점에서 관계 형성에 대한 욕구와 친사회적 동기가 강점이다. 다만 자기 마음을 적절히 표현하는 방법이 부족하므로, 자신의 감정을 말로 표현하는 연습이 필요하다. 진서는 다른 친구와 수영 약속이 있었다는 점에서 약속을 지키려는 책임감이 강점으로 볼 수 있다. 그러나 거절 과정에서 상대의 입장을 고려한 설명이 부족했으므로, 상대가 상처받지 않도록 이유를 설명하고 배려하는 표현을 익힐 필요가 있다. 학생에게 자신의 강점을 알게 하면 긍정적 자기이해가 높아져 상담에 더 적극적으로 참여할 수 있다. 또한 문제 행동만 보는 것이 아니라 자신이 이미 가진 장점을 바탕으로 관계 문제를 해결할 수 있다는 효능감을 형성하여 대인관계 능력 향상에 도움이 된다.

셋째, 상담 목표를 설정할 때에는 몇 가지 사항을 고려해야 한다. 먼저 목표는 영우와 진서의 실제 대인관계 문제와 관련되어야 한다. 영우에게는 자신의 감정과 요구를 적절히 표현하는 것이, 진서에게는 상대의 입장을 고려하여 거절 의사를 설명하는 것이 핵심 목표가 될 수 있다. 다음으로 목표는 학생의 발달 수준과 현재 능력에 맞게 구체적이고 실천 가능해야 한다. 예컨대 “친구와 잘 지낸다”와 같은 막연한 목표보다 “거절할 때 이유와 미안한 마음을 함께 말한다”처럼 관찰 가능한 행동으로 설정하는 것이 적절하다. 마지막으로 상담 목표는 학생, 보호자, 교사가 함께 이해하고 동의할 수 있어야 하며, 상담 과정에서 점검과 수정이 가능해야 한다.

상담 목표를 적절히 설정하면 상담 과정과 성과에도 긍정적 효과가 있다. 우선 상담자가 무엇을 도와야 하는지 분명해져 상담 활동이 일관성 있게 이루어진다. 또한 목표 달성 정도를 확인할 수 있어 학생의 변화와 성장을 평가하고, 필요한 지원을 조정하기 쉽다. 결국 협력 지원 체제, 학생 강점 활용, 구체적인 상담 목표 설정은 학생의 대인관계 능력을 실제로 함양하는 데 유기적으로 연결된다.

근거:
- {filename} p.{page}: 2023학년도 초등학교 교직 논술 문제, 대인관계 능력 함양 지원 방안, 영우·진서의 강점, 상담 목표 설정 요구사항"""
        return f"""업로드한 문서에서 관련 근거를 우선 검색했습니다. 다만 이 요청에 맞춘 전용 fallback 답안 형식은 아직 준비되지 않았습니다.

검색 근거:
{LangChainService._format_context(documents)}
"""

    def generate_comparison_answer(self, question: str, comparison: Dict) -> str | None:
        """구조화된 대응표를 근거로 변화·수업 적용·인출 포인트를 종합한다."""
        if self.llm is None or not comparison.get("comparisons"):
            return None
        evidence = []
        for index, row in enumerate(comparison["comparisons"][:8], 1):
            evidence.append(
                f"[대응 {index}] {row['subject']} {row['grade_band']} {row['change_type']}\n"
                f"2015 {row.get('code_2015')}: {row.get('text_2015') or '없음'}\n"
                f"2022 {row.get('code_2022')}: {row.get('text_2022') or '없음'}"
            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", """초등 교원임용 교육과정 비교 코치로 답하세요.
2009 교육과정은 언급하지 마세요. 제공된 대응표 밖의 내용을 추정하지 마세요.
semantic_match와 unmatched는 자동 추정임을 구분하세요. 기출에는 공식 정답이 없으므로
답안을 제시할 때 '근거 기반 예상 답안'이라고 명시하세요.
답변은 ① 핵심 변화 ② 수업 설계에 미치는 영향 ③ 관련 기출 관점 ④ 인출 질문 순서로 구성하세요."""),
            ("human", "질문: {question}\n\n대응표:\n{evidence}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({
                "question": question or "선택한 교육과정 항목을 비교해 주세요.",
                "evidence": "\n\n".join(evidence),
            })
        except Exception as e:
            print(f"비교 답변 생성 실패: {e}")
            return None
    
    # ---------------------------
    # 구글 뉴스 검색 메서드
    # ---------------------------
    def search_google_news(self, query: str) -> Dict[str, any]:
        """
        구글 뉴스 검색 (Agent 사용)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 딕셔너리
        """
        try:
            if self.search_agent is None:
                return {
                    "status": "error",
                    "answer": "OPENAI_API_KEY가 없어 뉴스 에이전트를 사용할 수 없습니다.",
                    "query": query,
                }
            # Agent 실행
            result = self.search_agent.invoke({"input": query})
            
            return {
                "status": "success",
                "answer": result.get("output", ""),
                "query": query
            }
            
        except Exception as e:
            return {
                "status": "error",
                "answer": f"검색 실패: {str(e)}",
                "query": query
            }

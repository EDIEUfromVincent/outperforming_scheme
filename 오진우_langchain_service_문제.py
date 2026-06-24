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
from typing import List, Dict
from pathlib import Path

# LangChain 관련 라이브러리
from langchain_community.document_loaders import PyPDFLoader      # PDF 파일 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 텍스트 분할기
from langchain_community.vectorstores import FAISS                # 벡터 저장소
from langchain_openai import OpenAIEmbeddings, ChatOpenAI         # OpenAI 모델
from langchain_core.prompts import ChatPromptTemplate             # 프롬프트 템플릿
from langchain_core.runnables import RunnablePassthrough          # 데이터 전달용
from langchain_core.output_parsers import StrOutputParser         # 출력 파서
from langchain.tools import tool                                  # 도구 데코레이터
from langchain.agents import create_tool_calling_agent, AgentExecutor  # Agent 관련
from langchain_teddynote.tools import GoogleNews                  # 구글 뉴스 도구


# ============================================
# 2. LangChainService 클래스 정의
# ============================================
class LangChainService:
    """PDF 문서 기반 QA 서비스 + 구글 뉴스 검색 Agent"""
    
    def __init__(self, openai_api_key: str, faiss_db_path: str = "faiss_db"):
        """
        서비스 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            faiss_db_path: FAISS DB 저장 경로
        """
        # API 키 환경변수 설정
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # OpenAI 임베딩 모델 설정 (텍스트를 벡터로 변환)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # ChatGPT 모델 설정
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # 사용할 모델
            temperature=0,              # 응답의 일관성 (0: 일관적, 1: 창의적)
            max_tokens=1024             # 최대 토큰 수
        )
        
        # FAISS DB 경로 저장
        self.faiss_db_path = faiss_db_path
        
        # 벡터 스토어, 검색기, 체인 초기화
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        # 구글 뉴스 검색 Agent 설정
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
        @tool
        def search_news(query: str) -> List[Dict[str, str]]:
            """Search Google News by input keyword"""
            news_tool = GoogleNews()
            return news_tool.search_by_keyword(query, k=5)
        
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
                search_type="similarity",
                search_kwargs={'k': 5}
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
        
        # 프롬프트 템플릿 정의
        template = ChatPromptTemplate.from_messages([
            ('system', '당신은 문서 내용을 기반으로 답변을 제공하는 전문 AI 어시스턴트입니다.'),
            ('system', '''아래 제공된 context를 주의깊게 읽고 사용자의 질문에 답변해주세요.
            
규칙:
1. context에서 관련된 정보를 찾아 자세히 설명하세요
2. 여러 부분에 걸쳐 있는 정보는 통합하여 설명하세요
3. context에 정보가 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요
4. 가능한 한 구체적이고 상세하게 답변하세요

Context:
{context}'''),
            ('human', '{question}')
        ])
        
        # 체인 구성: 데이터 → 프롬프트 → LLM → 문자열 출력
        data = {"question": RunnablePassthrough(), "context": self.retriever}
        self.chain = data | template | self.llm | StrOutputParser()
    
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
            # 1. PDF 파일 로드
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()  # 페이지별로 문서 로드
            
            # 2. 텍스트를 작은 청크로 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,    # 청크 크기 (글자 수)
                chunk_overlap=200   # 청크 간 겹침 (문맥 유지용)
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
            
            # 5. 검색기 및 체인 설정
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 5}
            )
            self._setup_chain()
            
            return {
                "status": "success",
                "message": f"PDF 처리 완료: {len(chunks)}개 청크 생성",
                "chunks_count": len(chunks),
                "pages_count": len(documents)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"PDF 처리 실패: {str(e)}"
            }
    
    # ---------------------------
    # 스트리밍 질의응답 메서드
    # ---------------------------
    def query_stream(self, question: str):
        """
        질문에 대한 답변 생성 (스트리밍 방식)
        
        Args:
            question: 사용자 질문
            
        Yields:
            답변 토큰 (한 글자씩)
        """
        # 체인이 설정되지 않은 경우
        if self.chain is None:
            yield "먼저 PDF 문서를 업로드해주세요."
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
    def get_retrieved_documents(self, question: str) -> List[Dict]:
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
            retrieved_docs = self.retriever.invoke(question)
            
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

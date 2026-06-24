"""
Streamlit 앱 - PDF 질의응답 + 뉴스 검색 UI
이 파일은 사용자 인터페이스를 담당합니다.
"""

# ============================================
# 1. 필요한 라이브러리 불러오기
# ============================================
import streamlit as st  # 웹 UI를 만들기 위한 라이브러리
import requests         # API 호출을 위한 라이브러리
from pathlib import Path

# ============================================
# 2. 기본 설정
# ============================================
# FastAPI 서버 주소 (main.py가 실행되는 주소)
API_URL = "http://localhost:8010"

# 페이지 기본 설정
st.set_page_config(
    page_title="PDF QA & News Search",  # 브라우저 탭에 표시될 제목
    page_icon="📚",                      # 브라우저 탭에 표시될 아이콘
    layout="wide"                        # 화면 전체 너비 사용
)

# 페이지 제목 표시
st.title("📚 PDF 문서 기반 질의응답 + 🔍 뉴스 검색 시스템")

# ============================================
# 3. 세션 상태 초기화
# 세션 상태: 페이지가 새로고침되어도 유지되는 데이터 저장소
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []       # 대화 기록 저장
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = [] # 검색된 문서 저장
if "search_results" not in st.session_state:
    st.session_state.search_results = [] # 뉴스 검색 결과 저장

# ============================================
# 4. 탭 생성 (PDF 질의응답 / 뉴스 검색)
# ============================================
tab1, tab2 = st.tabs(["📄 PDF 질의응답", "🔍 뉴스 검색"])


# ===========================
# 탭 1: PDF 질의응답
# ===========================
with tab1:
    
    # ---------------------------
    # 사이드바: 문서 업로드 영역
    # ---------------------------
    with st.sidebar:
        st.header("📤 문서 업로드")
    
        # 파일 업로드 위젯
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],  # PDF 파일만 허용
            help="PDF 파일을 업로드하면 자동으로 분석됩니다"
        )
        
        # 파일이 업로드되었을 때
        if uploaded_file is not None:
            if st.button("업로드 및 처리", use_container_width=True):
                with st.status("PDF 처리 중...", expanded=True) as status:
                    try:
                        st.write(f"🔗 서버에 연결 중... ({API_URL})")
                        
                        #########################################################
                        # 문제1:
                        # streamlit 파일 업로드 컴포넌트 사용
                        # fastapi로 띄워진 /upload-pdf 엔드포인트 호출
                        
                        
                        #########################################################
                        files= {
                            "file": (
                                uploaded_file.name,
                                uploaded_file,
                                "application/pdf",
                            )
                        }

                        response = requests.post(
                            f"{API_URL}/upload-pdf",
                            files = files,
                            timeout=60,
                        )

                        # 응답 처리
                        if response.status_code == 200:
                            result = response.json()
                            status.update(label="✅ 처리 완료", state="complete")
                            st.success(result["message"])
                            st.info(f"📄 페이지 수: {result['pages_count']}")
                            st.info(f"📦 청크 수: {result['chunks_count']}")
                        else:
                            status.update(label="❌ 처리 실패", state="error")
                            error_detail = response.json().get('detail', '알 수 없는 오류')
                            st.error(f"오류: {error_detail}")
                            
                    except requests.exceptions.ConnectionError:
                        status.update(label="❌ 처리 실패", state="error")
                        st.error("⚠️ FastAPI 서버에 연결할 수 없습니다!")
                        st.error("서버가 실행 중인지 확인하세요: python main.py")
                    except requests.exceptions.Timeout:
                        status.update(label="❌ 처리 실패", state="error")
                        st.error("⏱️ 요청 시간 초과!")
                    except Exception as e:
                        status.update(label="❌ 처리 실패", state="error")
                        st.error(f"오류: {str(e)}")
        
        st.divider()  # 구분선
        
        # 대화 기록 초기화 버튼
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.retrieved_docs = []
            st.rerun()  # 페이지 새로고침
        
        # 사용 방법 안내
        with st.expander("ℹ️ 사용 방법"):
            st.markdown("""
            **PDF 업로드 및 처리**
            - 사이드바에서 PDF 파일을 선택합니다
            - 업로드 및 처리 버튼을 클릭합니다
            
            **질문하기**
            - 메인 화면 하단의 입력창에 질문을 입력합니다
            - AI가 문서를 기반으로 답변을 생성합니다
            """)
    
    # ---------------------------
    # 메인 영역: 대화 인터페이스
    # ---------------------------
    st.header("💬 질문하기")
    
    # 이전 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 질문 입력창
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # 응답을 표시할 빈 공간
            full_response = ""
            
            try:
                #########################################################
                response = requests.post(
                    json={"question": prompt},
                    stream=True,
                    timeout=60,
                )

                if response.status_code == 200:
                    for chunk in response.iter_content(decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                
                else:
                    full_response = f"❌ error: {response.status_code} - {response.text}"
                    message_placeholder.markdown(full_response)
                #########################################################
            
            except Exception as e:
                full_response = f"❌ 오류 발생: {str(e)}"
                message_placeholder.markdown(full_response)
            
            # 응답 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 관련 문서 검색
        try:
            doc_response = requests.post(
                f"{API_URL}/documents",
                json={"question": prompt}
            )
            
            if doc_response.status_code == 200:
                st.session_state.retrieved_docs = doc_response.json().get("documents", [])
        except Exception as e:
            st.error(f"❌ 문서 검색 실패: {str(e)}")
    
    # ---------------------------
    # 검색된 문서 표시
    # ---------------------------
    if st.session_state.retrieved_docs:
        st.divider()
        st.header("📚 검색된 관련 문서")
        
        for i, doc in enumerate(st.session_state.retrieved_docs, 1):
            with st.expander(f"📄 문서 {i}", expanded=(i==1)):
                st.markdown("**내용:**")
                st.text(doc["content"])
                
                # 메타데이터 표시
                if doc.get("metadata"):
                    st.markdown("**메타데이터:**")
                    metadata = doc["metadata"]
                    cols = st.columns(3)
                    if "page" in metadata:
                        cols[0].metric("페이지", metadata["page"] + 1)
                    if "source" in metadata:
                        cols[1].text(f"출처: {Path(metadata['source']).name}")


# ===========================
# 탭 2: 뉴스 검색
# ===========================
with tab2:
    st.header("🔍 구글 뉴스 검색")
    
    # 검색 UI
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "검색어를 입력하세요",
            placeholder="예: AI 투자, 삼성전자, ChatGPT",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 검색", use_container_width=True, type="primary")
    
    # 검색 실행
    if search_button and search_query:
        with st.spinner("🔍 뉴스 검색 중..."):
            try:
                # 검색 요청
                response = requests.post(
                    f"{API_URL}/search",
                    json={"query": search_query}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["status"] == "success":
                        st.success(f"✅ '{search_query}' 검색 완료!")
                        st.markdown("### 📰 검색 결과")
                        st.markdown(result["answer"])
                        st.session_state.search_results = result
                    else:
                        st.error(f"❌ {result['answer']}")
                else:
                    st.error(f"❌ 검색 실패: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
    
    # 이전 검색 결과 표시
    elif st.session_state.search_results and "answer" in st.session_state.search_results:
        st.markdown("### 📰 이전 검색 결과")
        st.info(f"검색어: {st.session_state.search_results.get('query', '')}")
        st.markdown(st.session_state.search_results["answer"])
    
    # 안내 메시지
    else:
        st.info("💡 검색어를 입력하고 검색 버튼을 클릭하세요")
        st.markdown("""
        ### 📌 사용 예시
        - **AI 관련**: "AI 투자", "ChatGPT", "OpenAI"
        """)
    
    # 검색 기록 초기화
    st.divider()
    if st.button("🗑️ 검색 기록 초기화", use_container_width=True):
        st.session_state.search_results = []
        st.rerun()

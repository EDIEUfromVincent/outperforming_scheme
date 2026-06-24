"""
Streamlit 앱 - PDF 질의응답 + 뉴스 검색 UI
이 파일은 사용자 인터페이스를 담당합니다.
"""

# ============================================
# 1. 필요한 라이브러리 불러오기
# ============================================
import streamlit as st  # 웹 UI를 만들기 위한 라이브러리
import requests         # API 호출을 위한 라이브러리
import pandas as pd
import os
from pathlib import Path

# ============================================
# 2. 기본 설정
# ============================================
# FastAPI 서버 주소 (main.py가 실행되는 주소)
API_URL = os.getenv("API_URL", "http://localhost:8010")
PROJECT3_API_URL = os.getenv("PROJECT3_API_URL", "http://localhost:8020")
DEFAULT_FILTERS = {
    "subjects": ["과학", "국어", "도덕", "미술", "사회", "수학", "실과", "영어", "음악", "체육", "통합교과"],
    "grade_bands": ["1-2", "3-4", "5-6"],
    "domains": [f"{index:02d}" for index in range(1, 18)],
}

# 페이지 기본 설정
st.set_page_config(
    page_title="초등임용 교육과정 코치",  # 브라우저 탭에 표시될 제목
    page_icon="📚",                      # 브라우저 탭에 표시될 아이콘
    layout="wide"                        # 화면 전체 너비 사용
)

# 페이지 제목 표시
st.title("📚 초등임용 교육과정·수업설계 코치")
st.caption(
    "2015·2022 개정 교육과정, 평가기준·성취수준, 기출문제 근거를 연결해 "
    "초등임용고시 수험생의 생생한 인출과 수업 설계를 돕는 도구입니다."
)

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
if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None
if "active_document_name" not in st.session_state:
    st.session_state.active_document_name = None


def get_backend_status() -> dict:
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            data = health_response.json()
            return {
                "ok": bool(data.get("features", {}).get("document_aware_query")),
                "version": data.get("version"),
                "message": "최신 백엔드 연결됨",
            }
    except Exception:
        pass
    try:
        openapi_response = requests.get(f"{API_URL}/openapi.json", timeout=5)
        if openapi_response.status_code == 200:
            paths = openapi_response.json().get("paths", {})
            schemas = openapi_response.json().get("components", {}).get("schemas", {})
            query_props = schemas.get("QueryRequest", {}).get("properties", {})
            ok = "/upload-document" in paths and "document_id" in query_props
            return {
                "ok": ok,
                "version": "unknown",
                "message": "백엔드가 최신입니다." if ok else "백엔드가 구버전입니다. python main.py를 재시작하세요.",
            }
    except Exception as exc:
        return {"ok": False, "version": None, "message": f"백엔드 연결 실패: {exc}"}
    return {"ok": False, "version": None, "message": "백엔드 상태를 확인하지 못했습니다."}


backend_status = get_backend_status()
if not backend_status["ok"]:
    st.error(
        f"⚠️ FastAPI 백엔드가 현재 Streamlit 코드와 맞지 않습니다. {backend_status['message']} "
        "이 상태에서는 새로 올린 PDF가 질문 답변에 반영되지 않을 수 있습니다."
    )
else:
    st.caption(f"백엔드 상태: {backend_status['message']} · {backend_status.get('version')}")

# ============================================
# 4. 탭 생성 (교육과정 QA / 비교 / 수업-임용 코치 / 뉴스 검색)
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 교육과정 QA",
    "🔄 2015↔2022 비교",
    "🧭 수업-임용 통합 코치",
    "📈 학습 대시보드",
    "📌 임용 공지·소식",
])


# ===========================
# 탭 1: PDF 질의응답
# ===========================
with tab1:
    
    # ---------------------------
    # 사이드바: 문서 업로드 영역
    # ---------------------------
    with st.sidebar:
        st.header("📤 문서 업로드")

        if st.button("📚 교육과정 자료 일괄 색인", use_container_width=True):
            with st.spinner("교육과정·평가기준 자료를 분석하고 있습니다..."):
                try:
                    library_response = requests.post(
                        f"{API_URL}/ingest-library", timeout=3600
                    )
                    if library_response.status_code == 200:
                        library_result = library_response.json()
                        st.success(
                            f"신규 {library_result['indexed_count']}개 · "
                            f"기존 {library_result['skipped_count']}개 · "
                            f"실패 {library_result['failed_count']}개"
                        )
                    else:
                        st.error(library_response.json().get("detail", "일괄 색인 실패"))
                except Exception as e:
                    st.error(f"일괄 색인 오류: {e}")

        st.divider()
        try:
            uploaded_docs_response = requests.get(f"{API_URL}/uploaded-documents", timeout=30)
            uploaded_docs = (
                uploaded_docs_response.json().get("documents", [])
                if uploaded_docs_response.status_code == 200
                else []
            )
        except Exception:
            uploaded_docs = []

        valid_uploaded_docs = [doc for doc in uploaded_docs if doc.get("document_id")]
        if valid_uploaded_docs:
            labels = [doc["filename"] for doc in valid_uploaded_docs]
            default_index = 0
            if st.session_state.active_document_id:
                for idx, doc in enumerate(valid_uploaded_docs):
                    if doc["document_id"] == st.session_state.active_document_id:
                        default_index = idx
                        break
            selected_label = st.selectbox(
                "우선 답변 문서",
                labels,
                index=default_index,
                help="질문 답변 시 이 문서를 최우선 근거로 사용합니다.",
            )
            selected_doc = valid_uploaded_docs[labels.index(selected_label)]
            st.session_state.active_document_id = selected_doc["document_id"]
            st.session_state.active_document_name = selected_doc["filename"]
            st.caption(f"선택됨: {selected_doc['pages_count']}쪽")

        st.divider()
        
        # 파일 업로드 위젯
        uploaded_file = st.file_uploader(
            "PDF/TXT/MD 파일을 선택하세요",
            type=['pdf', 'txt', 'md'],
            help="PDF, 교직논술, 인강 요약본(txt/md)을 업로드하면 자동으로 학습데이터에 추가됩니다"
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
                        # fastapi로 띄워진 /upload-document 엔드포인트 호출
                        
                        # 파일 업로드
                        suffix = Path(uploaded_file.name).suffix.lower()
                        mime_type = "application/pdf" if suffix == ".pdf" else "text/plain"
                        files = {"file": (uploaded_file.name, uploaded_file, mime_type)}
                        st.write(f"📤 파일 업로드 중: {uploaded_file.name}")
                        
                        response = requests.post(
                            f"{API_URL}/upload-document", 
                            files=files,
                            timeout=60  # 1분 타임아웃
                        )
                        if response.status_code == 404 and suffix == ".pdf":
                            st.warning("새 업로드 API가 현재 FastAPI 서버에 아직 반영되지 않아 기존 PDF 업로드 API로 다시 시도합니다.")
                            uploaded_file.seek(0)
                            files = {"file": (uploaded_file.name, uploaded_file, mime_type)}
                            response = requests.post(
                                f"{API_URL}/upload-pdf",
                                files=files,
                                timeout=60,
                            )
                        
                        st.write(f"📥 응답 상태 코드: {response.status_code}")
                        
                        #########################################################
                        
                        # 응답 처리
                        if response.status_code == 200:
                            result = response.json()
                            status.update(label="✅ 처리 완료", state="complete")
                            st.success(result["message"])
                            st.info(f"📄 페이지 수: {result['pages_count']}")
                            st.info(f"📦 청크 수: {result['chunks_count']}")
                            st.session_state.active_document_id = result.get("document_id")
                            st.session_state.active_document_name = safe_name = result.get("filename", uploaded_file.name)
                            st.success(f"이제 질문은 우선 이 문서 기준으로 답합니다: {safe_name}")
                            if result.get("document_type") == "lecture_note":
                                note = result.get("lecture_note") or {}
                                learning_data = result.get("learning_data") or {}
                                st.success("인강 요약본으로 인식하여 학습자료 DB에 추가했습니다.")
                                st.info(
                                    f"교과: {note.get('subject') or '미분류'} · "
                                    f"주제: {', '.join(note.get('topics', [])[:6]) or '자동 추출 없음'} · "
                                    f"기출연도: {', '.join(map(str, note.get('exam_years', []))) or '없음'}"
                                )
                                if learning_data.get("message"):
                                    st.caption(learning_data["message"])
                            methods = ", ".join(result.get("extraction_methods", []))
                            if methods:
                                st.info(f"🔎 추출 방식: {methods}")
                            for warning in result.get("warnings", []):
                                st.warning(warning)
                        else:
                            status.update(label="❌ 처리 실패", state="error")
                            try:
                                error_detail = response.json().get('detail', '알 수 없는 오류')
                            except Exception:
                                error_detail = response.text or '알 수 없는 오류'
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
    if st.session_state.active_document_id:
        st.caption(f"현재 우선 문서: {st.session_state.active_document_name}")
    
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
                # 문제2-(1):
                # streamlit 스트리밍 응답 컴포넌트 사용
                # fastapi로 띄워진 /query/stream 엔드포인트 호출
                # 에러 헨들링
                
                # 스트리밍 응답 요청
                response = requests.post(
                    f"{API_URL}/query/stream",
                    json={
                        "question": prompt,
                        "document_id": st.session_state.active_document_id,
                    },
                    stream=True
                )
                
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                else:
                    full_response = f"❌ 오류: {response.status_code}"
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
                json={
                    "question": prompt,
                    "document_id": st.session_state.active_document_id,
                }
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
# 탭 2: 교육과정 비교
# ===========================
with tab2:
    st.header("🔄 2015↔2022 개정 교육과정 비교")
    st.info(
        "초등임용고시 수험생 관점에서 성취기준 변화가 수업 목표, 학생 활동, "
        "교사 지원, 평가 증거에 어떤 영향을 주는지 확인합니다. "
        "2015년 기출은 OCR 보정 파서로 검색용 문항 chunk를 복구했으며, 정밀 인용은 추가 검수가 필요합니다."
    )
    try:
        filter_response = requests.get(f"{API_URL}/comparison/filters", timeout=30)
        filters = filter_response.json() if filter_response.status_code == 200 else {}
    except Exception:
        filters = DEFAULT_FILTERS
    filters = {**DEFAULT_FILTERS, **filters}

    col_subject, col_grade, col_domain = st.columns(3)
    with col_subject:
        selected_subject = st.selectbox("교과", ["전체"] + filters.get("subjects", []))
    with col_grade:
        selected_grade = st.selectbox("학년군", ["전체"] + filters.get("grade_bands", []))
    with col_domain:
        selected_domain = st.selectbox("영역 코드", ["전체"] + filters.get("domains", []))
    comparison_query = st.text_input(
        "비교할 개념이나 수업 주제",
        placeholder="예: 토의·토론, 과정 중심 평가, 분수의 나눗셈",
    )
    if st.button("교육과정 비교", type="primary", use_container_width=True):
        payload = {
            "subject": None if selected_subject == "전체" else selected_subject,
            "grade_band": None if selected_grade == "전체" else selected_grade,
            "domain": None if selected_domain == "전체" else selected_domain,
            "query": comparison_query,
            "limit": 12,
        }
        try:
            response = requests.post(f"{API_URL}/comparison", json=payload, timeout=120)
            if response.status_code != 200:
                st.error(response.json().get("detail", "비교 검색 실패"))
            else:
                result = response.json()
                st.caption(result.get("notice", ""))
                if result.get("answer"):
                    st.markdown(result["answer"])
                    st.divider()
                for row in result.get("comparisons", []):
                    title = (
                        f"{row['subject']} {row['grade_band']} · "
                        f"{row.get('code_2015') or '신설'} → {row.get('code_2022') or '삭제'}"
                    )
                    with st.expander(title):
                        st.markdown(f"**변화 유형:** `{row['change_type']}`")
                        st.markdown("**2015 개정**")
                        st.write(row.get("text_2015") or "해당 대응 없음")
                        if row.get("source_2015"):
                            st.caption(f"{row['source_2015']} · p.{row['page_2015']}")
                        st.markdown("**2022 개정**")
                        st.write(row.get("text_2022") or "해당 대응 없음")
                        if row.get("source_2022"):
                            st.caption(f"{row['source_2022']} · p.{row['page_2022']}")
                if result.get("related_exams"):
                    st.subheader("관련 기출")
                    for exam in result["related_exams"]:
                        meta = exam["metadata"]
                        with st.expander(
                            f"{meta.get('exam_year', '')} {meta.get('form', '')} "
                            f"{meta.get('subject', '')} {meta.get('question_number', '')}번"
                        ):
                            st.write(exam["content"])
        except Exception as e:
            st.error(f"비교 검색 오류: {e}")


# ===========================
# 탭 3: 수업-임용 통합 코치
# ===========================
with tab3:
    st.header("🧭 총론·평가기준 감독 수업-임용 통합 코치")
    st.info(
        "총괄 에이전트가 총론과 평가기준·성취수준을 감독 기준으로 삼고, "
        "교과 전문 에이전트와 1~6학년 학생 에이전트 관점을 묶어 수업 설계와 임용 인출을 연결합니다."
    )

    try:
        coach_filter_response = requests.get(f"{API_URL}/comparison/filters", timeout=30)
        coach_filters = coach_filter_response.json() if coach_filter_response.status_code == 200 else {}
    except Exception:
        coach_filters = DEFAULT_FILTERS
    coach_filters = {**DEFAULT_FILTERS, **coach_filters}

    coach_col1, coach_col2 = st.columns(2)
    with coach_col1:
        coach_subject = st.selectbox(
            "담당 교과",
            ["자동 추론"] + coach_filters.get("subjects", []),
            key="coach_subject",
        )
    with coach_col2:
        coach_grade = st.selectbox(
            "담당 학년",
            ["자동 추론", 1, 2, 3, 4, 5, 6],
            key="coach_grade",
        )

    coach_query = st.text_area(
        "수업과 임용 공부를 함께 묻기",
        placeholder="예: 내일 5학년 수학 분수의 나눗셈 수업을 해야 하는데, 임용 관점에서 성취기준 변화와 기출 포인트까지 같이 공부하고 싶어.",
        height=120,
    )

    if st.button("총괄 에이전트로 수업-임용 연결하기", type="primary", use_container_width=True):
        payload = {
            "query": coach_query,
            "subject": None if coach_subject == "자동 추론" else coach_subject,
            "grade": None if coach_grade == "자동 추론" else int(coach_grade),
            "limit": 6,
        }
        if not coach_query.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            try:
                response = requests.post(f"{API_URL}/lesson-coach", json=payload, timeout=180)
                if response.status_code != 200:
                    st.error(response.json().get("detail", "수업-임용 코치 호출 실패"))
                else:
                    result = response.json()
                    route = result.get("route", {})
                    st.caption(result.get("notice", ""))
                    route_cols = st.columns(4)
                    route_cols[0].metric("교과", route.get("subject") or "미지정")
                    route_cols[1].metric("학년", f"{route.get('grade')}학년" if route.get("grade") else "미지정")
                    route_cols[2].metric("학년군", route.get("grade_band") or "미지정")
                    route_cols[3].metric("의도", route.get("intent") or "통합")

                    st.markdown(result.get("answer", "답변을 생성하지 못했습니다."))

                    with st.expander("총괄 에이전트 감사 체크리스트"):
                        audit = result.get("audit", {})
                        st.write(f"통과 여부: {audit.get('passed')}")
                        for label, passed in audit.get("checks", {}).items():
                            st.write(f"{'✅' if passed else '⚠️'} {label}")

                    with st.expander("총론·평가기준 근거"):
                        for doc in result.get("governance_docs", []):
                            st.markdown(f"**{doc.get('label', '근거')}**")
                            st.write(doc.get("content", ""))

                    with st.expander("교과 전문 에이전트 결과"):
                        subject_result = result.get("subject_agent", {})
                        for point in subject_result.get("summary_points", []):
                            st.write(f"- {point}")
                        st.write(f"검수 필요 대응 수: {subject_result.get('review_required_count', 0)}")

                    with st.expander("학년 학생 에이전트 관점"):
                        grade_result = result.get("grade_student_agent", {})
                        st.write(f"초점: {grade_result.get('focus')}")
                        st.write("예상 오개념/장벽:")
                        for item in grade_result.get("misconceptions", []):
                            st.write(f"- {item}")
                        st.write("교사 지원:")
                        for item in grade_result.get("support", []):
                            st.write(f"- {item}")
            except Exception as e:
                st.error(f"수업-임용 코치 오류: {e}")


# ===========================
# 탭 4: 학습 대시보드
# ===========================
with tab4:
    st.header("📈 에빙하우스 복습·이동평균 학습 대시보드")
    st.info(
        "퀴즈/인출 결과를 SQLite에 저장하고, 일간 정답률·5일/20일 이동평균선·하향 돌파 취약 성취기준·오늘의 복습 목록을 확인합니다."
    )

    dash_col1, dash_col2 = st.columns([1, 1])
    with dash_col1:
        if st.button("데모 학습 기록 생성", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/learning/seed-demo", timeout=30)
                if response.status_code == 200:
                    st.success(response.json().get("message", "데모 데이터 생성 완료"))
                else:
                    st.error(response.json().get("detail", "데모 데이터 생성 실패"))
            except Exception as e:
                st.error(f"데모 데이터 오류: {e}")
    with dash_col2:
        dashboard_days = st.slider("조회 기간", min_value=14, max_value=180, value=60, step=1)
        aggregate_period = st.radio("장기 흐름", ["주간", "월간"], horizontal=True)

    with st.expander("퀴즈/인출 결과 직접 저장", expanded=False):
        attempt_col1, attempt_col2, attempt_col3 = st.columns(3)
        with attempt_col1:
            attempt_code = st.text_input("성취기준 코드", value="6수01-11")
            attempt_subject = st.selectbox("교과", DEFAULT_FILTERS["subjects"], index=5, key="attempt_subject")
        with attempt_col2:
            attempt_grade_band = st.selectbox("학년군", DEFAULT_FILTERS["grade_bands"], index=2, key="attempt_grade")
            attempt_correct = st.radio("결과", ["맞힘", "헷갈림/틀림"], horizontal=True)
        with attempt_col3:
            attempt_confidence = st.slider("확신도", 1, 5, 3)
            attempt_time = st.number_input("풀이 시간(초)", min_value=0, value=90, step=10)
        attempt_question = st.text_area("질문/문제", placeholder="예: 분수의 나눗셈 원리를 설명하시오.", height=80)
        attempt_answer = st.text_area("내 답변", placeholder="간단히 기록", height=80)
        if st.button("풀이 결과 저장", use_container_width=True):
            payload = {
                "standard_code": attempt_code,
                "subject": attempt_subject,
                "grade_band": attempt_grade_band,
                "is_correct": attempt_correct == "맞힘",
                "confidence": attempt_confidence,
                "time_spent_sec": int(attempt_time),
                "question_text": attempt_question,
                "user_answer": attempt_answer,
                "source_type": "manual",
            }
            try:
                response = requests.post(f"{API_URL}/learning/attempt", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"저장 완료 · attempt_id={result.get('attempt_id')} · 복습 주기={result.get('scheduled_reviews')}")
                else:
                    st.error(response.json().get("detail", "저장 실패"))
            except Exception as e:
                st.error(f"저장 오류: {e}")

    try:
        metric_response = requests.get(f"{API_URL}/learning/metrics", params={"days": dashboard_days}, timeout=30)
        metrics = metric_response.json().get("metrics", []) if metric_response.status_code == 200 else []
    except Exception:
        metrics = []

    if metrics:
        metric_df = pd.DataFrame(metrics)
        chart_df = metric_df[["day", "accuracy", "ma_5", "ma_20"]].copy()
        chart_df["day"] = pd.to_datetime(chart_df["day"])
        chart_df = chart_df.set_index("day")
        st.subheader("일간 정답률과 5일/20일 이동평균선")
        st.line_chart(chart_df)

        latest = metric_df.iloc[-1]
        metric_cols = st.columns(4)
        metric_cols[0].metric("오늘 시도", int(latest["attempts"]))
        metric_cols[1].metric("오늘 정답률", f"{latest['accuracy'] * 100:.1f}%")
        metric_cols[2].metric("5일 MA", f"{latest['ma_5'] * 100:.1f}%")
        metric_cols[3].metric("20일 MA", f"{latest['ma_20'] * 100:.1f}%")

        try:
            period_code = "M" if aggregate_period == "월간" else "W"
            aggregate_response = requests.get(
                f"{API_URL}/learning/metrics/aggregate",
                params={"days": dashboard_days, "period": period_code},
                timeout=30,
            )
            aggregate_metrics = (
                aggregate_response.json().get("metrics", [])
                if aggregate_response.status_code == 200
                else []
            )
        except Exception:
            aggregate_metrics = []
        if aggregate_metrics:
            aggregate_df = pd.DataFrame(aggregate_metrics)
            aggregate_chart_df = aggregate_df[["period", "accuracy", "ma_5", "ma_20"]].set_index("period")
            st.subheader(f"{aggregate_period} 정답률과 장기 이동평균선")
            st.line_chart(aggregate_chart_df)
    else:
        st.warning("아직 학습 기록이 없습니다. 데모 학습 기록을 생성하거나 풀이 결과를 저장해 보세요.")

    weak_col, due_col = st.columns(2)
    with weak_col:
        st.subheader("하향 돌파·취약 성취기준")
        try:
            weak_response = requests.get(f"{API_URL}/learning/weak-standards", params={"limit": 8}, timeout=30)
            weak_rows = weak_response.json().get("weak_standards", []) if weak_response.status_code == 200 else []
        except Exception:
            weak_rows = []
        if not weak_rows:
            st.caption("취약 성취기준을 계산할 기록이 아직 부족합니다.")
        for index, row in enumerate(weak_rows):
            label = (
                f"{row.get('subject', '')} {row.get('grade_band', '')} {row['standard_code']} "
                f"· 정답률 {row['accuracy'] * 100:.1f}% · 약점점수 {row['weak_score']}"
            )
            with st.expander(label, expanded=index == 0):
                if row.get("downside_cross"):
                    st.warning("5일 이동평균선이 20일 이동평균선 아래로 내려간 하향 돌파 상태입니다.")
                st.write(row.get("standard_text") or "성취기준 원문 확인 필요")
                if st.button(f"변형 문제 생성: {row['standard_code']}", key=f"variant_{index}"):
                    payload = {
                        "standard_code": row["standard_code"],
                        "subject": row.get("subject"),
                        "grade_band": row.get("grade_band"),
                        "weakness_note": "최근 정답률 또는 이동평균선이 낮아진 성취기준",
                    }
                    try:
                        response = requests.post(f"{API_URL}/learning/generate-variant", json=payload, timeout=30)
                        if response.status_code == 200:
                            st.markdown(response.json().get("quiz_text", ""))
                        else:
                            st.error(response.json().get("detail", "변형 문제 생성 실패"))
                    except Exception as e:
                        st.error(f"변형 문제 생성 오류: {e}")

    with due_col:
        st.subheader("오늘의 에빙하우스 복습")
        try:
            due_response = requests.get(f"{API_URL}/learning/review-due", params={"limit": 10}, timeout=30)
            due_rows = due_response.json().get("due_reviews", []) if due_response.status_code == 200 else []
        except Exception:
            due_rows = []
        if not due_rows:
            st.caption("오늘 복습 예정인 성취기준이 없습니다.")
        for row in due_rows:
            with st.expander(
                f"{row.get('subject', '')} {row.get('grade_band', '')} {row['standard_code']} · {row['due_count']}회 복습"
            ):
                st.write(f"복습 예정일: {row.get('next_review_at')}")
                st.write(f"주기: {row.get('intervals')}일")
                st.write(row.get("standard_text") or "성취기준 원문 확인 필요")
                if st.button(f"복습 완료 처리: {row['standard_code']}", key=f"complete_review_{row['standard_code']}"):
                    payload = {"standard_code": row["standard_code"]}
                    try:
                        response = requests.post(f"{API_URL}/learning/review-complete", json=payload, timeout=30)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"{result.get('completed_reviews', 0)}개 복습 일정을 완료 처리했습니다.")
                        else:
                            st.error(response.json().get("detail", "복습 완료 처리 실패"))
                    except Exception as e:
                        st.error(f"복습 완료 처리 오류: {e}")


# ===========================
# 탭 5: 임용 공지·교육과정 소식
# ===========================
with tab5:
    st.header("📌 임용 공지·교육과정 소식")
    st.info(
        "구글 뉴스 검색 대신 Project 3의 임용 브리핑 서비스를 연결합니다. "
        "기출 1문제, 예비문제 1문제, 공식 공지 후보를 이메일 본문으로 생성합니다."
    )

    st.markdown("""
    Project 3 실행 방법:

    ```bash
    cd /Users/vincent/Downloads/project/project3
    /opt/anaconda3/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8020
    ```

    별도 화면:

    ```bash
    streamlit run app.py --server.port 8502
    ```
    """)

    recipient = st.text_input("브리핑 수신자", value="ohjinwoo9696@gmail.com")
    include_regions = st.checkbox("17개 시·도교육청 공고 후보까지 포함", value=False)

    col_notice, col_digest = st.columns(2)
    with col_notice:
        if st.button("공식 공지 후보 수집", use_container_width=True):
            try:
                response = requests.get(
                    f"{PROJECT3_API_URL}/notices",
                    params={"include_regions": include_regions},
                    timeout=80,
                )
                response.raise_for_status()
                data = response.json()
                st.success(f"{data.get('count', 0)}개 후보 수집")
                for item in data.get("items", [])[:12]:
                    with st.expander(f"[{item.get('source')}] {item.get('title')}"):
                        st.write(f"분류: {item.get('category')}")
                        st.write(f"링크: {item.get('url')}")
                for warning in data.get("warnings", []):
                    st.warning(warning)
            except Exception as e:
                st.error(f"Project 3 API에 연결할 수 없습니다: {e}")

    with col_digest:
        if st.button("오늘의 메일 미리보기", use_container_width=True):
            try:
                response = requests.post(
                    f"{PROJECT3_API_URL}/digest/preview",
                    json={
                        "recipient": recipient,
                        "include_regions": include_regions,
                        "send": False,
                    },
                    timeout=100,
                )
                response.raise_for_status()
                digest = response.json()
                st.success(digest.get("subject"))
                st.text_area("메일 본문", value=digest.get("body", ""), height=520)
            except Exception as e:
                st.error(f"Project 3 API에 연결할 수 없습니다: {e}")

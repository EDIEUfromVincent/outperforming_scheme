"""Project 3 Streamlit UI: 임용 공지·교육뉴스·일일문제 메일러."""

from __future__ import annotations

import streamlit as st
import requests


API_URL = "http://localhost:8020"
DEFAULT_RECIPIENT = "ohjinwoo9696@gmail.com"


st.set_page_config(page_title="초등임용 일일 브리핑", page_icon="✉️", layout="wide")
st.title("✉️ 초등임용 일일 브리핑 메일러")
st.caption("기출 1문제, 예비문제 1문제, 임용 공지·교육과정 소식을 하루 한 통으로 묶습니다.")

tab1, tab2, tab3 = st.tabs(["📌 공지 수집", "🧪 메일 미리보기", "📨 발송 설정"])


with tab1:
    st.header("임용 공지·교육과정 소식")
    include_regions = st.checkbox("17개 시·도교육청까지 포함", value=False)
    if st.button("공지 후보 수집", use_container_width=True):
        try:
            response = requests.get(
                f"{API_URL}/notices",
                params={"include_regions": include_regions},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            st.success(f"{data.get('source_count')}개 출처에서 {data.get('count')}개 후보를 수집했습니다.")
            for item in data.get("items", []):
                with st.expander(f"[{item.get('source')}] {item.get('title')}"):
                    st.write(f"분류: {item.get('category')}")
                    st.write(f"키워드: {', '.join(item.get('matched_keywords', []))}")
                    st.link_button("원문 열기", item.get("url"))
            for warning in data.get("warnings", []):
                st.warning(warning)
        except Exception as exc:
            st.error(f"수집 실패: {exc}")


with tab2:
    st.header("오늘의 메일 미리보기")
    recipient = st.text_input("수신자", value=DEFAULT_RECIPIENT)
    include_regions_preview = st.checkbox("메일에도 시·도교육청 포함", value=False, key="preview_regions")
    seed = st.text_input("테스트 seed", value="", help="비워두면 오늘 날짜 기준으로 매일 같은 문제가 선택됩니다.")
    if st.button("브리핑 생성", use_container_width=True):
        try:
            response = requests.post(
                f"{API_URL}/digest/preview",
                json={
                    "recipient": recipient,
                    "include_regions": include_regions_preview,
                    "send": False,
                    "seed": seed or None,
                },
                timeout=80,
            )
            response.raise_for_status()
            digest = response.json()
            st.subheader(digest.get("subject"))
            st.text_area("메일 본문", value=digest.get("body", ""), height=600)
        except Exception as exc:
            st.error(f"브리핑 생성 실패: {exc}")


with tab3:
    st.header("Gmail 발송 설정")
    st.info(
        "실제 발송에는 .env에 EMAIL_ADDRESS와 EMAIL_PASSWORD(Gmail 앱 비밀번호)가 필요합니다. "
        "발송 버튼을 눌러도 API의 send=true 요청이 가야만 메일이 나갑니다."
    )
    try:
        status = requests.get(f"{API_URL}/email/status", timeout=20).json()
        st.json(status)
    except Exception as exc:
        st.error(f"상태 확인 실패: {exc}")

    send_recipient = st.text_input("발송 수신자", value=DEFAULT_RECIPIENT, key="send_recipient")
    send_regions = st.checkbox("발송 메일에 시·도교육청 포함", value=False, key="send_regions")
    confirm = st.checkbox("위 수신자로 오늘의 브리핑 메일을 실제 발송합니다.")
    if st.button("실제 이메일 발송", type="primary", use_container_width=True, disabled=not confirm):
        try:
            response = requests.post(
                f"{API_URL}/digest/email",
                json={
                    "recipient": send_recipient,
                    "include_regions": send_regions,
                    "send": True,
                },
                timeout=100,
            )
            if response.status_code == 200:
                st.success("이메일 발송 완료")
                st.json(response.json())
            else:
                st.error(response.json().get("detail", "발송 실패"))
        except Exception as exc:
            st.error(f"발송 실패: {exc}")

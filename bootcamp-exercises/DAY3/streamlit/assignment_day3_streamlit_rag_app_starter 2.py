"""Day 3 과제: Streamlit RAG 챗봇 앱 완성하기.

실행:
    streamlit run DAY3/streamlit/assignment_day3_streamlit_rag_app_starter.py

목표:
    CLI RAG 챗봇을 Streamlit 웹 앱으로 바꿉니다.
"""

from __future__ import annotations

import streamlit as st

from common import generate_answer, make_sources_dataframe, retrieve_documents


st.set_page_config(
    page_title="Day 3 RAG 챗봇",
    page_icon="RAG",
    layout="wide",
)

st.title("Day 3 문서 기반 RAG 챗봇")
st.write("PostgreSQL/pgvector에 저장된 문서 chunk를 검색해서 답변합니다.")

with st.sidebar:
    st.header("검색 설정")

    top_k = st.slider("검색할 chunk 개수", min_value=1, max_value=8, value=4)
    show_sources = st.checkbox("검색 출처 표 표시", value=True)

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("질문을 입력하세요")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("문서를 검색하고 답변을 생성하는 중입니다..."):
            results = retrieve_documents(question, top_k=top_k)
            answer = generate_answer(question, results)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_sources = results

        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        st.error("오류가 발생했습니다.")
        st.exception(e)

if show_sources and st.session_state.last_sources:
    st.subheader("검색된 출처")
    source_df = make_sources_dataframe(st.session_state.last_sources)
    st.dataframe(source_df, use_container_width=True, hide_index=True)

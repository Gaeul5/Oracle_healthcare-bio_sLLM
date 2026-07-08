"""Day 3 도전 과제: Streamlit RAG 품질 점검 대시보드.

실행:
    streamlit run DAY3/streamlit/assignment_day3_streamlit_rag_dashboard_starter.py

목표:
    답변뿐 아니라 검색된 chunk, distance, context를 함께 확인합니다.
"""

from __future__ import annotations

import streamlit as st

from common import (
    build_answer_markdown,
    format_context,
    generate_answer,
    make_sources_dataframe,
    retrieve_documents,
)


st.set_page_config(
    page_title="RAG 품질 점검 대시보드",
    page_icon="RAG",
    layout="wide",
)

st.title("Streamlit RAG 품질 점검 대시보드")
st.write("답변만 보지 말고, 검색된 chunk와 distance, context를 함께 확인합니다.")

with st.sidebar:
    st.header("검색 설정")

    top_k = st.slider("검색할 chunk 개수 TOP_K", min_value=1, max_value=10, value=4)
    distance_threshold = st.slider(
        "distance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
    )
    show_context = st.checkbox("LLM context 보기", value=False)
    show_preview = st.checkbox("검색 preview 보기", value=True)

    st.caption("distance는 낮을수록 질문과 더 가깝습니다.")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_results" not in st.session_state:
    st.session_state.last_results = []

question = st.chat_input("질문을 입력하세요")

if question:
    st.session_state.last_question = question

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("검색 중입니다..."):
            results = retrieve_documents(
                question,
                top_k=top_k,
                distance_threshold=distance_threshold,
            )

        st.session_state.last_results = results

        if not results:
            answer = (
                "검색 조건에 맞는 문서를 찾지 못했습니다. "
                "distance threshold를 높이거나 질문을 더 구체적으로 바꿔 보세요."
            )
        else:
            with st.spinner("답변을 생성하는 중입니다..."):
                answer = generate_answer(question, results)

        st.session_state.last_answer = answer

        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        st.error("오류가 발생했습니다.")
        st.exception(e)

if st.session_state.last_results:
    st.subheader("검색된 출처")
    source_df = make_sources_dataframe(st.session_state.last_results)
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    if show_preview:
        st.subheader("검색 결과 preview")
        for i, row in enumerate(st.session_state.last_results, start=1):
            page = row["page_number"] if row["page_number"] is not None else "-"
            title = f"{i}. {row['file_name']} / p.{page} / distance={float(row['distance']):.4f}"
            with st.expander(title):
                st.write(row["content"])

    if show_context:
        st.subheader("LLM에게 전달된 context")
        context = format_context(st.session_state.last_results)
        st.text_area("context", value=context, height=360)

if st.session_state.last_answer:
    markdown = build_answer_markdown(
        st.session_state.last_question,
        st.session_state.last_answer,
        st.session_state.last_results,
    )
    st.download_button(
        "답변 Markdown 다운로드",
        data=markdown,
        file_name="rag_answer.md",
        mime="text/markdown",
    )

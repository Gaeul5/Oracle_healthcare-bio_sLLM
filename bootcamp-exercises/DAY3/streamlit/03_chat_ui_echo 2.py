"""Streamlit 채팅 UI의 최소 구조."""

import streamlit as st

st.set_page_config(page_title="Echo Chat", page_icon="💬")

st.title("💬 Streamlit Echo Chat")
st.write("입력한 문장을 assistant가 그대로 따라 말하는 가장 작은 채팅 앱입니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지를 다시 그립니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("메시지를 입력하세요")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = f"제가 들은 말은: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

with st.sidebar:
    st.header("대화 관리")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
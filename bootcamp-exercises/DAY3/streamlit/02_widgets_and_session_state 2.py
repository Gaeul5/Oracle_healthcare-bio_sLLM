"""위젯과 session_state 이해하기."""

import streamlit as st

st.set_page_config(page_title="Widgets & State", page_icon="🎛️")

st.title("🎛️ Streamlit 위젯과 session_state")
st.write("Streamlit 앱은 사용자가 입력하거나 버튼을 누를 때마다 위에서 아래로 다시 실행됩니다.")

if "count" not in st.session_state:
    st.session_state.count = 0

with st.sidebar:
    st.header("설정")
    top_k = st.slider("검색할 chunk 개수 TOP_K", min_value=1, max_value=8, value=4)
    show_debug = st.checkbox("검색 결과 preview 보기", value=True)

st.subheader("현재 설정")
st.write(f"TOP_K: {top_k}")
st.write(f"검색 결과 preview 보기: {show_debug}")

st.subheader("session_state 카운터")
st.write(f"현재 count: {st.session_state.count}")

col1, col2 = st.columns(2)
with col1:
    if st.button("count 증가"):
        st.session_state.count += 1
with col2:
    if st.button("count 초기화"):
        st.session_state.count = 0

st.info(
    "일반 변수는 rerun 때 다시 만들어질 수 있습니다. "
    "대화 기록처럼 유지해야 하는 값은 st.session_state에 저장합니다."
)
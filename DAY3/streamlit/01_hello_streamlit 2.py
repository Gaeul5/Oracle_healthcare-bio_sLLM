"""가장 작은 Streamlit 앱."""

import streamlit as st

st.set_page_config(page_title="Hello Streamlit", page_icon="👋")

st.title("👋 Hello, Streamlit")
st.write("이 파일은 Streamlit 앱이 정상적으로 실행되는지 확인하는 첫 번째 예제입니다.")

name = st.text_input("이름을 입력하세요", value="수강생")

if st.button("인사하기"):
    st.success(f"안녕하세요, {name}님! 이제 Python 코드가 웹 화면으로 보입니다.")

st.divider()
st.markdown("""
### 기억할 점

- Streamlit 파일은 `python 파일명.py`가 아니라 `streamlit run 파일명.py`로 실행합니다.
- `st.title()`, `st.write()`, `st.button()` 같은 함수가 화면 요소가 됩니다.
""")
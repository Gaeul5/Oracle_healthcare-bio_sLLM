# FastAPI LangChain Chatbot

Python 3.12, FastAPI, LangChain LCEL, PydanticOutputParser, Runnable.stream을 사용한 웹 챗봇 예제입니다.

## 실행

```bash
cd fastapi_chatbot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_API_KEY"
uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 됩니다.

## 구조

```text
fastapi_chatbot/
  app/
    main.py
    schemas.py
    services/
      chat_service.py
    static/
      css/style.css
      js/app.js
    templates/
      index.html
```

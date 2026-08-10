# LangChain RAG Chatbot Starter

이 프로젝트는 LangChain 강의 과제용 스타터입니다.
서버와 화면은 실행되지만, 핵심 기능은 TODO로 남겨져 있습니다.

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_server.py
```

macOS/Linux에서는 가상환경 실행 명령이 다릅니다.

```bash
source .venv/bin/activate
cp .env.example .env
python run_server.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:8000
```

## 학생이 구현할 핵심 기능

- JavaScript fetch로 API 호출
- FastAPI endpoint 구현
- 문서 로드
- Text Splitter로 chunk 분리
- OpenAI Embeddings로 embedding 생성
- 기존 PostgreSQL 테이블에 저장
- pgvector 유사도 검색
- LangChain Prompt + ChatOpenAI 답변 생성
- 답변 아래 출처 표시

## 사용하는 기존 DB 테이블

- rag_documents
- rag_chunks
- rag_embeddings

새 테이블을 만들지 않습니다.

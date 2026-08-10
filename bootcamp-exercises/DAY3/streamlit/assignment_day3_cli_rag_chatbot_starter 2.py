"""Day 3 과제 1: CLI RAG 챗봇.

실행 예시:
    python DAY3/streamlit/assignment_day3_cli_rag_chatbot_starter.py

완성 조건:
- 질문을 입력받는다.
- 질문을 embedding한다.
- rag_embeddings 테이블에서 가까운 chunk를 검색한다.
- 검색된 chunk를 context로 만든다.
- LLM이 context 기반으로 답변한다.
- 답변 아래에 출처와 distance를 출력한다.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from psycopg.rows import dict_row

load_dotenv()

TOP_K = 4

SYSTEM_PROMPT = """너는 문서 기반으로 답변하는 RAG 챗봇입니다.

규칙:
1. 반드시 제공된 context를 우선해서 답변하세요.
2. context에 없는 내용은 추측하지 말고 모른다고 말하세요.
3. 답변은 한국어로 작성하세요.
4. 가능하면 핵심 내용을 번호 목록으로 정리하세요.
"""

TEST_QUESTIONS = [
    "중국 AI 플러스 정책의 핵심 영역은 무엇인가요?",
    "Qwen3-Next의 특징을 요약해줘.",
    "교육 분야에서 AI는 어떻게 활용되고 있나요?",
]


def env(name: str, default: str = "") -> str:
    """환경변수 값을 읽고 앞뒤 공백을 제거합니다."""
    return os.getenv(name, default).strip()


def connect_db():
    """PostgreSQL에 연결합니다."""
    return psycopg.connect(
        host=env("POSTGRES_HOST", "localhost"),
        port=int(env("POSTGRES_PORT", "5432")),
        dbname=env("POSTGRES_DB"),
        user=env("POSTGRES_USER"),
        password=env("POSTGRES_PASSWORD"),
        row_factory=dict_row,
    )


def to_pgvector(vector: list[float]) -> str:
    """Python list[float]를 pgvector가 읽을 수 있는 문자열로 바꿉니다."""
    return "[" + ",".join(str(x) for x in vector) + "]"


def embed_question(question: str) -> tuple[str, str]:
    """질문을 embedding하고 DB 검색용 pgvector 문자열로 변환합니다."""
    embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    query_vector = embeddings.embed_query(question)
    return to_pgvector(query_vector), embedding_model


def search_chunks(question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """질문과 가까운 chunk를 rag_embeddings 테이블에서 검색합니다."""
    query_vector_text, embedding_model = embed_question(question)

    sql = """
        SELECT
            d.file_name,
            d.file_type,
            c.chunk_index,
            c.page_number,
            c.content,
            e.embedding <=> %s::vector AS distance
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        JOIN rag_documents d ON c.document_id = d.id
        WHERE e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector_text, embedding_model, query_vector_text, top_k))
            return [dict(row) for row in cur.fetchall()]


def format_context(results: list[dict[str, Any]]) -> str:
    """검색된 chunk를 LLM에게 전달할 context 문자열로 만듭니다."""
    context_parts: list[str] = []
    for i, row in enumerate(results, start=1):
        page = row["page_number"] if row["page_number"] is not None else "-"
        context_parts.append(
            f"[문서 {i} / {row['file_name']} / page={page} / distance={float(row['distance']):.4f}]\n"
            f"{row['content']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(question: str, results: list[dict[str, Any]]) -> str:
    """검색 결과를 context로 넣어 LLM 답변을 생성합니다."""
    if not results:
        return "문서에서 충분한 근거를 찾지 못했습니다. DB 적재 상태를 확인하거나 질문을 바꿔 보세요."

    context = format_context(results)
    user_prompt = f"""아래 context만 참고해서 질문에 답변하세요.

[context]
{context}

[질문]
{question}
"""

    llm = ChatOpenAI(model=env("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def print_sources(results: list[dict[str, Any]]) -> None:
    """답변 아래에 출처와 distance를 출력합니다."""
    print("\n[검색된 출처]")
    if not results:
        print("- 검색된 출처가 없습니다.")
        return

    for i, row in enumerate(results, start=1):
        page = row["page_number"] if row["page_number"] is not None else "-"
        print(
            f"{i}. {row['file_name']} ({row['file_type']}) "
            f"/ page={page} / chunk={row['chunk_index']} / distance={float(row['distance']):.4f}"
        )


def print_test_questions() -> None:
    print("[테스트 질문]")
    for question in TEST_QUESTIONS:
        print(f"- {question}")
    print()


def main() -> None:
    print("Day 3 과제 1: CLI RAG 챗봇")
    print("종료하려면 quit, exit, 종료 중 하나를 입력하세요.\n")
    print_test_questions()

    while True:
        question = input("질문 > ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit"} or question == "종료":
            print("챗봇을 종료합니다.")
            break

        try:
            results = search_chunks(question)
            answer = generate_answer(question, results)
        except Exception as e:
            print("\n오류가 발생했습니다.")
            print(e)
            print(".env, OpenAI API 키, PostgreSQL 실행 상태, rag_* 테이블 적재 상태를 확인하세요.\n")
            continue

        print("\n" + "=" * 80)
        print("답변")
        print("=" * 80)
        print(answer)
        print_sources(results)
        print()


if __name__ == "__main__":
    main()

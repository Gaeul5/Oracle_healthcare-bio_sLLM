"""공통 RAG 함수 모음.

이 파일은 Day 3에서 만든 DB 구조를 그대로 사용합니다.
Streamlit 파일에서는 이 파일의 함수를 import해서 사용합니다.

사용 테이블:
- rag_documents
- rag_chunks
- rag_embeddings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from psycopg.rows import dict_row

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

SYSTEM_PROMPT = """너는 문서 기반으로 답변하는 RAG 챗봇입니다.

규칙:
1. 반드시 제공된 context를 우선해서 답변하세요.
2. context에 없는 내용은 추측하지 말고 모른다고 말하세요.
3. 답변은 한국어로 작성하세요.
4. 가능하면 답변을 문단 또는 번호 목록으로 정리하세요.
"""


def env(name: str, default: str = "") -> str:
    """환경변수 값을 읽고 앞뒤 공백을 제거합니다."""
    return os.getenv(name, default).strip()


def get_int_env(name: str, default: int) -> int:
    raw = env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def get_float_env(name: str, default: float) -> float:
    raw = env(name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def connect_db():
    """PostgreSQL에 연결합니다.

    row_factory=dict_row를 사용하면 SELECT 결과를 dict처럼 사용할 수 있습니다.
    예: row["file_name"], row["content"]
    """
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


def get_embeddings() -> OpenAIEmbeddings:
    embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=embedding_model)


def get_llm(temperature: float = 0) -> ChatOpenAI:
    model = env("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


def get_db_counts() -> dict[str, int]:
    """Day 3 데이터가 들어 있는지 확인하기 위한 카운트 함수입니다."""
    counts: dict[str, int] = {}
    with connect_db() as conn:
        with conn.cursor() as cur:
            for table in ["rag_documents", "rag_chunks", "rag_embeddings"]:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                counts[table] = int(cur.fetchone()["count"])
    return counts


def retrieve_documents(
    question: str,
    top_k: int = 4,
    distance_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """질문을 embedding하고 DB에서 가까운 chunk를 검색합니다.

    Parameters
    ----------
    question:
        사용자의 질문입니다.
    top_k:
        검색할 chunk 개수입니다.
    distance_threshold:
        None이면 필터링하지 않습니다. 숫자를 넣으면 distance가 이 값 이하인 결과만 반환합니다.

    Returns
    -------
    list[dict]
        file_name, chunk_index, page_number, content, distance를 가진 dict 리스트입니다.
    """
    embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = get_embeddings()

    query_vector = embeddings.embed_query(question)
    query_vector_text = to_pgvector(query_vector)

    sql = """
        SELECT
            d.file_name,
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
            results = [dict(row) for row in cur.fetchall()]

    if distance_threshold is not None:
        results = [row for row in results if float(row["distance"]) <= distance_threshold]

    return results


def format_context(results: list[dict[str, Any]]) -> str:
    """검색 결과를 LLM에게 전달할 context 문자열로 변환합니다."""
    context_parts: list[str] = []
    for i, row in enumerate(results, start=1):
        context_parts.append(
            f"[문서 {i} / {row['file_name']} / p.{row['page_number']} / distance={float(row['distance']):.4f}]\n"
            f"{row['content']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(question: str, results: list[dict[str, Any]]) -> str:
    """검색 결과를 context로 만들어 LLM 답변을 생성합니다."""
    if not results:
        return "문서에서 충분한 근거를 찾지 못했습니다. 질문을 더 구체적으로 바꾸거나 DB 적재 상태를 확인하세요."

    context = format_context(results)
    user_prompt = f"""아래 context만 참고해서 질문에 답변하세요.

[context]
{context}

[질문]
{question}
"""

    llm = get_llm(temperature=0)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return response.content


def make_sources_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """검색 결과를 Streamlit의 st.dataframe으로 보여주기 좋은 형태로 바꿉니다."""
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(results, start=1):
        content = row.get("content") or ""
        rows.append(
            {
                "rank": rank,
                "file_name": row.get("file_name"),
                "page_number": row.get("page_number"),
                "chunk_index": row.get("chunk_index"),
                "distance": round(float(row.get("distance", 0)), 4),
                "preview": content[:160].replace("\n", " "),
            }
        )
    return pd.DataFrame(rows)


def build_answer_markdown(question: str, answer: str, results: list[dict[str, Any]]) -> str:
    """다운로드용 Markdown 문자열을 만듭니다."""
    lines = ["# RAG 답변", "", "## 질문", "", question, "", "## 답변", "", answer, "", "## 검색된 출처", ""]
    if not results:
        lines.append("검색된 출처가 없습니다.")
        return "\n".join(lines)

    lines.append("| 순위 | 파일명 | 페이지 | distance | preview |")
    lines.append("|---:|---|---:|---:|---|")
    for i, row in enumerate(results, start=1):
        preview = (row.get("content") or "")[:80].replace("\n", " ").replace("|", " ")
        lines.append(
            f"| {i} | {row.get('file_name')} | {row.get('page_number')} | {float(row.get('distance', 0)):.4f} | {preview} |"
        )
    return "\n".join(lines)
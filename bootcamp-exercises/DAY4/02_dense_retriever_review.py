import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from psycopg2.extras import RealDictCursor

load_dotenv()


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"환경변수 {name} 값이 없습니다. .env 파일을 확인하세요.")
    return value or ""


def connect_db():
    return psycopg2.connect(
        host=env("POSTGRES_HOST", "localhost"),
        port=env("POSTGRES_PORT", "5432"),
        dbname=env("POSTGRES_DB", required=True),
        user=env("POSTGRES_USER", required=True),
        password=env("POSTGRES_PASSWORD", required=True),
    )


def get_embedding_model_name() -> str:
    return env("OPENAI_EMBEDDING_MODEL") or env("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=get_embedding_model_name())


def get_top_k(default: int = 4) -> int:
    raw = env("TOP_K", str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in values) + "]"


def fetch_all(conn, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def dense_search(conn, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
    embeddings = get_embeddings()
    query_vector = to_pgvector(embeddings.embed_query(query))
    embedding_model = get_embedding_model_name()

    sql = """
        SELECT
            c.id AS chunk_id,
            d.file_name,
            c.page_number,
            c.content,
            e.embedding <=> %s::vector AS distance,
            'dense' AS retriever_type
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        JOIN rag_documents d ON c.document_id = d.id
        WHERE e.embedding_model = %s
        ORDER BY distance ASC
        LIMIT %s;
    """
    rows = fetch_all(conn, sql, (query_vector, embedding_model, top_k))
    for row in rows:
        row["score_label"] = "distance"
        row["score"] = row.get("distance")
        row["matched_by"] = ["dense"]
    return rows


def preview(text: str | None, length: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:length] + ("..." if len(compact) > length else "")


def print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("검색 결과가 없습니다. 질문을 더 짧거나 명확한 키워드로 바꿔보세요.")
        return
    for i, row in enumerate(results, start=1):
        label = row.get("score_label", "score")
        score = row.get("score")
        score_text = f"{label}={score:.4f}" if isinstance(score, (int, float)) else f"{label}={score}"
        matched = "+".join(row.get("matched_by") or [row.get("retriever_type", "")])
        print(f"\n[{i}] {row.get('file_name')} / p.{row.get('page_number')} / {matched} / {score_text}")
        print(preview(row.get("content")))

QUESTION = "AI 에이전트와 관련된 산업 동향을 알려줘."


def main():
    conn = connect_db()
    top_k = get_top_k(4)

    print("Dense Retriever")
    print("질문:", QUESTION)
    print("설명: 질문을 embedding하고, DB의 chunk embedding과 distance를 계산합니다.")

    results = dense_search(conn, QUESTION, top_k=top_k)
    print_results(results)

    conn.close()


if __name__ == "__main__":
    main()
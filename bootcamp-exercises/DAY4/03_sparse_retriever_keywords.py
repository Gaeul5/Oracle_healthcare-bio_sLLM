import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
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


def get_top_k(default: int = 4) -> int:
    raw = env("TOP_K", str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def fetch_all(conn, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def sparse_search(conn, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    sql = """
        SELECT
            c.id AS chunk_id,
            d.file_name,
            c.page_number,
            c.content,
            ts_rank_cd(
                to_tsvector('simple', coalesce(c.content, '')),
                websearch_to_tsquery('simple', %s)
            ) AS sparse_score,
            'sparse' AS retriever_type
        FROM rag_chunks c
        JOIN rag_documents d ON c.document_id = d.id
        WHERE to_tsvector('simple', coalesce(c.content, ''))
              @@ websearch_to_tsquery('simple', %s)
        ORDER BY sparse_score DESC
        LIMIT %s;
    """
    rows = fetch_all(conn, sql, (query, query, top_k))
    for row in rows:
        row["score_label"] = "sparse_score"
        row["score"] = row.get("sparse_score")
        row["matched_by"] = ["sparse"]
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

QUESTION = "Qwen3-Next 효율성"


def main():
    conn = connect_db()
    top_k = get_top_k(4)

    print("Sparse Retriever")
    print("질문/키워드:", QUESTION)
    print("설명: embedding을 만들지 않고, rag_chunks.content에서 키워드를 직접 찾습니다.")

    results = sparse_search(conn, QUESTION, top_k=top_k)
    print_results(results)

    conn.close()


if __name__ == "__main__":
    main()
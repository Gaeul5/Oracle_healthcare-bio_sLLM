import os

import psycopg2
from dotenv import load_dotenv

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

REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "EMBEDDING_MODEL",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]


def main():
    print("[1] .env 값 확인")
    for key in REQUIRED_ENV:
        value = env(key, "")
        print(f"- {key}: {'OK' if value else '비어 있음'}")

    print("\n[2] DB 연결 확인")
    conn = connect_db()
    print("- DB 연결 성공")

    with conn.cursor() as cur:
        print("\n[3] pgvector extension 확인")
        cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        print("- vector extension:", "OK" if cur.fetchone()[0] else "없음")

        print("\n[4] Day 3 테이블 데이터 확인")
        for table in ["rag_documents", "rag_chunks", "rag_embeddings"]:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            print(f"- {table}: {cur.fetchone()[0]} rows")

        print("\n[5] embedding_model 확인")
        cur.execute("""
            SELECT embedding_model, COUNT(*)
            FROM rag_embeddings
            GROUP BY embedding_model
            ORDER BY COUNT(*) DESC;
        """)
        rows = cur.fetchall()
        current = get_embedding_model_name()
        if not rows:
            print("- 저장된 embedding이 없습니다. Day 3 PDF 적재를 먼저 완료하세요.")
        for model, count in rows:
            mark = "<-- 현재 .env" if model == current else ""
            print(f"- {model}: {count} rows {mark}")

    conn.close()
    print("\nDay 4 준비 확인이 끝났습니다.")


if __name__ == "__main__":
    main()
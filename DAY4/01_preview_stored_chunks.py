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


def preview(text: str | None, length: int = 220) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:length] + ("..." if len(compact) > length else "")


def main():
    conn = connect_db()
    with conn.cursor() as cur:
        print("[1] 문서별 chunk 개수")
        cur.execute("""
            SELECT d.id, d.file_name, d.file_type, COUNT(c.id) AS chunk_count
            FROM rag_documents d
            LEFT JOIN rag_chunks c ON d.id = c.document_id
            GROUP BY d.id, d.file_name, d.file_type
            ORDER BY d.id;
        """)
        for doc_id, file_name, file_type, count in cur.fetchall():
            print(f"- document_id={doc_id} / {file_type} / chunks={count} / {file_name}")

        print("\n[2] 저장된 chunk preview")
        cur.execute("""
            SELECT c.id, d.file_name, c.page_number, c.content
            FROM rag_chunks c
            JOIN rag_documents d ON c.document_id = d.id
            ORDER BY c.id
            LIMIT 8;
        """)
        for chunk_id, file_name, page_number, content in cur.fetchall():
            print(f"\nchunk_id={chunk_id} / {file_name} / p.{page_number}")
            print(preview(content))

    conn.close()


if __name__ == "__main__":
    main()
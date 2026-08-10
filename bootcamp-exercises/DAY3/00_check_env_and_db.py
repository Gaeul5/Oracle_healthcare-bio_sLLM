from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row

load_dotenv()


def env(name, default=""):
    return os.getenv(name, default).strip()


print("[1] .env 확인")
print("OPENAI_API_KEY:", "있음" if env("OPENAI_API_KEY") and env("OPENAI_API_KEY") != "OPENAI_API_KEY" else "없음")
print("OPENAI_MODEL:", env("OPENAI_MODEL", "gpt-4o-mini"))
print("OPENAI_EMBEDDING_MODEL:", env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
print("POSTGRES_HOST:", env("POSTGRES_HOST"))
print("POSTGRES_PORT:", env("POSTGRES_PORT"))
print("POSTGRES_DB:", env("POSTGRES_DB"))
print("POSTGRES_USER:", env("POSTGRES_USER"))

print("\n[2] DB 연결 확인")
try:
    conn = psycopg.connect(
        host=env("POSTGRES_HOST", "localhost"),
        port=int(env("POSTGRES_PORT", "5432")),
        dbname=env("POSTGRES_DB"),
        user=env("POSTGRES_USER"),
        password=env("POSTGRES_PASSWORD"),
        row_factory=dict_row,
    )
    print("DB 연결 성공")
except Exception as e:
    print("DB 연결 실패")
    print(e)
    raise SystemExit

with conn.cursor() as cur:
    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS ok")
    print("pgvector extension:", "OK" if cur.fetchone()["ok"] else "없음")

    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = {row["table_name"] for row in cur.fetchall()}

required = ["rag_documents", "rag_chunks", "rag_embeddings"]
print("\n[3] 실습 테이블 확인")
for table in required:
    print(table, "OK" if table in tables else "없음")

missing = [table for table in required if table not in tables]
if missing:
    print("\n테이블이 없습니다. 먼저 실행하세요:")
    print("psql -U <user> -d <db> -f sql/day3_schema.sql")
else:
    print("\n준비 완료. 다음 단계: python 01_preview_data.py")

conn.close()
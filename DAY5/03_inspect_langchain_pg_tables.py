from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "day5_pgvector_tool_demo"

db_config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "testdb"),
    "user": os.getenv("POSTGRES_USER", "test"),
    "password": os.getenv("POSTGRES_PASSWORD", "5748"),
}

print("=" * 80)
print("03. LangChain PGVector 테이블 확인")
print("=" * 80)

with psycopg.connect(**db_config) as conn:
    with conn.cursor() as cur:
        print("\n[테이블 목록]")
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('langchain_pg_collection', 'langchain_pg_embedding')
            ORDER BY table_name;
        """)
        for row in cur.fetchall():
            print(f"- {row[0]}")

        print("\n[collection 목록]")
        cur.execute("SELECT uuid, name FROM langchain_pg_collection ORDER BY name;")
        for uuid_value, name in cur.fetchall():
            print(f"- uuid: {uuid_value} | name: {name}")

        print("\n[collection별 embedding row 수]")
        cur.execute("""
            SELECT c.name, COUNT(e.id) AS embedding_count
            FROM langchain_pg_collection c
            LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
            GROUP BY c.uuid, c.name
            ORDER BY c.name;
        """)
        for name, embedding_count in cur.fetchall():
            print(f"- {name}: {embedding_count} rows")

        print(f"\n[현재 collection 예시 row]")
        cur.execute("""
            SELECT e.id, LEFT(e.document, 120), e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
            LIMIT 1;
        """, (COLLECTION_NAME,))
        row = cur.fetchone()
        if row:
            id_value, preview, metadata = row
            print(f"- id: {id_value}")
            print(f"- document preview: {preview}")
            print(f"- metadata: {metadata}")
        else:
            print("- 예시 row가 없습니다. 01번 파일을 먼저 실행하세요.")

print("\n다음 단계: python 04_tool_make_and_practice.py")
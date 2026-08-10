from __future__ import annotations

import sys

from common import connect_db

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("02. LangChain PGVector 테이블 확인")
print("- langchain_pg_collection")
print("- langchain_pg_embedding")
print("=" * 80)


def table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s);", (table_name,))
    return cur.fetchone()[0] is not None


def print_columns(cur, table_name: str) -> None:
    cur.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """,
        (table_name,),
    )
    rows = cur.fetchall()
    print(f"\n[{table_name} 컬럼]")
    for column_name, data_type in rows:
        print(f"- {column_name}: {data_type}")


with connect_db() as conn:
    with conn.cursor() as cur:
        for table in ["langchain_pg_collection", "langchain_pg_embedding"]:
            if not table_exists(cur, table):
                print(f"{table} 테이블이 아직 없습니다.")
                print("먼저 01_pgvector_from_documents_dense.py를 실행하세요.")
                raise SystemExit
            print_columns(cur, table)

        print("\n[collection별 embedding row 수]")
        cur.execute(
            """
            SELECT
                c.name,
                c.uuid,
                COUNT(e.id) AS embedding_count
            FROM langchain_pg_collection c
            LEFT JOIN langchain_pg_embedding e
                ON e.collection_id = c.uuid
            GROUP BY c.name, c.uuid
            ORDER BY c.name;
            """
        )
        for name, uuid, embedding_count in cur.fetchall():
            print(f"- collection_name={name} / uuid={uuid} / embedding_count={embedding_count}")

        print("\n[embedding 샘플 3개]")
        cur.execute(
            """
            SELECT
                c.name,
                e.document,
                e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c
                ON e.collection_id = c.uuid
            ORDER BY e.id
            LIMIT 3;
            """
        )
        for idx, (name, document, cmetadata) in enumerate(cur.fetchall(), 1):
            preview = (document or "")[:160].replace("\n", " ")
            print(f"\n[{idx}] collection={name}")
            print(f"document preview: {preview}...")
            print(f"metadata: {cmetadata}")
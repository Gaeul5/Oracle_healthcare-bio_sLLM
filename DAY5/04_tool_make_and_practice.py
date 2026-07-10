from __future__ import annotations

import json
import os
from datetime import datetime

import psycopg
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

COLLECTION_NAME = "day5_pgvector_tool_demo"
TOP_K = 3


def get_db_config() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "testdb"),
        "user": os.getenv("POSTGRES_USER", "test"),
        "password": os.getenv("POSTGRES_PASSWORD", "5748"),
    }


def get_connection_string() -> str:
    return PGVector.connection_string_from_db_params(
        driver="psycopg",
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "testdb"),
        user=os.getenv("POSTGRES_USER", "test"),
        password=os.getenv("POSTGRES_PASSWORD", "5748"),
    )


def get_vectorstore() -> PGVector:
    embeddings_model = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return PGVector(
        embeddings=embeddings_model,
        collection_name=COLLECTION_NAME,
        connection=get_connection_string(),
        use_jsonb=True,
    )


@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다. 사용자가 현재 시간이나 날짜를 물어볼 때 사용합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_documents(query: str, top_k: int = TOP_K) -> str:
    """질문과 의미적으로 유사한 문서 chunk를 PGVector에서 검색합니다. 등록된 문서 내용을 근거로 답해야 할 때 사용합니다."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    payload = []
    for idx, doc in enumerate(docs, 1):
        payload.append(
            {
                "rank": idx,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "알 수 없음"),
                "page": doc.metadata.get("page", "알 수 없음"),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def get_collection_stats() -> str:
    """현재 PGVector collection의 embedding row 개수와 source 개수를 반환합니다. collection 상태나 문서 저장 여부를 확인할 때 사용합니다."""
    with psycopg.connect(**get_db_config()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.name AS collection_name,
                    COUNT(e.id) AS embedding_count,
                    COUNT(DISTINCT e.cmetadata->>'source') AS distinct_source_count
                FROM langchain_pg_collection c
                JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                WHERE c.name = %s
                GROUP BY c.name
                """,
                (COLLECTION_NAME,),
            )
            row = cur.fetchone()

    if row is None:
        return f"collection_name={COLLECTION_NAME}, embedding_count=0, distinct_source_count=0"

    collection_name, embedding_count, distinct_source_count = row
    return (
        f"collection_name={collection_name}, "
        f"embedding_count={embedding_count}, "
        f"distinct_source_count={distinct_source_count}"
    )


@tool
def list_collections() -> str:
    """PGVector에 저장된 collection 목록을 보여줍니다. 어떤 collection이 저장되어 있는지 확인할 때 사용합니다."""
    with psycopg.connect(**get_db_config()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM langchain_pg_collection ORDER BY name")
            names = [row[0] for row in cur.fetchall()]

    return json.dumps(names, ensure_ascii=False, indent=2)


@tool
def get_chunk_detail(chunk_id: str) -> str:
    """langchain_pg_embedding의 특정 id row를 자세히 조회합니다. 특정 chunk의 원문과 metadata를 확인할 때 사용합니다."""
    with psycopg.connect(**get_db_config()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document, cmetadata FROM langchain_pg_embedding WHERE id = %s",
                (chunk_id,),
            )
            row = cur.fetchone()

    if row is None:
        return json.dumps(
            {"error": f"chunk_id={chunk_id}를 찾을 수 없습니다."},
            ensure_ascii=False,
            indent=2,
        )

    document, cmetadata = row
    return json.dumps(
        {"id": chunk_id, "document": document, "metadata": cmetadata},
        ensure_ascii=False,
        indent=2,
    )


print("=" * 80)
print("04. Tool 만들기와 직접 실행")
print("=" * 80)

print("\n[1] get_current_time")
print(get_current_time.invoke({}))

print("\n[2] search_documents")
print(search_documents.invoke({"query": "생성형 AI 기술 동향", "top_k": 2}))

print("\n[3] get_collection_stats")
try:
    print(get_collection_stats.invoke({}))
except NotImplementedError as e:
    print(e)

print("\n[4] list_collections")
try:
    print(list_collections.invoke({}))
except NotImplementedError as e:
    print(e)

print("\n[5] get_chunk_detail")
with psycopg.connect(**get_db_config()) as _conn:
    with _conn.cursor() as _cur:
        _cur.execute(
            "SELECT id FROM langchain_pg_embedding e "
            "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name = %s LIMIT 1",
            (COLLECTION_NAME,),
        )
        _sample = _cur.fetchone()

if _sample:
    print(get_chunk_detail.invoke({"chunk_id": str(_sample[0])}))
else:
    print(f"collection_name={COLLECTION_NAME}에 저장된 chunk가 없습니다.")

print("\n다음 단계: python 05_manual_tool_calling_once.py")
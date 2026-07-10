from __future__ import annotations

import json
import os
from datetime import datetime

import psycopg
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
def get_current_time() -> str:
    """현재 시간을 반환합니다. 사용자가 현재 시간이나 날짜를 물어볼 때 사용합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_collection_stats() -> str:
    """현재 PGVector collection의 embedding row 개수와 source 개수를 반환합니다. collection 상태나 문서 저장 여부를 확인할 때 사용합니다."""
    sql = """
        SELECT
            c.name,
            COUNT(e.id) AS embedding_count,
            COUNT(DISTINCT COALESCE(e.cmetadata->>'source', '')) AS source_count
        FROM langchain_pg_collection c
        LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
        WHERE c.name = %s
        GROUP BY c.uuid, c.name;
    """
    with psycopg.connect(**get_db_config()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (COLLECTION_NAME,))
            row = cur.fetchone()

    if not row:
        return f"collection '{COLLECTION_NAME}'을 찾지 못했습니다."

    name, embedding_count, source_count = row
    return (
        f"collection_name={name}, embedding_count={embedding_count}, "
        f"distinct_source_count={source_count}"
    )


@tool
def list_collections() -> str:
    """PGVector에 저장된 collection 목록을 보여줍니다. 어떤 collection이 저장되어 있는지 확인할 때 사용합니다."""
    sql = "SELECT name FROM langchain_pg_collection ORDER BY name;"
    with psycopg.connect(**get_db_config()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    names = [row[0] for row in rows]
    return json.dumps(names, ensure_ascii=False, indent=2)


@tool
def get_chunk_detail(chunk_id: str) -> str:
    """langchain_pg_embedding의 특정 id row를 자세히 조회합니다. 특정 chunk의 원문과 metadata를 확인할 때 사용합니다."""
    sql = """
        SELECT c.name, e.id, e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE e.id::text = %s;
    """
    with psycopg.connect(**get_db_config()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (chunk_id,))
            row = cur.fetchone()

    if not row:
        return f"id={chunk_id} row를 찾지 못했습니다."

    collection_name, id_value, document, metadata = row
    payload = {
        "collection_name": collection_name,
        "id": str(id_value),
        "document": document,
        "metadata": metadata,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


TOOLS = [
    search_documents,
    get_current_time,
    get_collection_stats,
    list_collections,
    get_chunk_detail,
]


def get_used_tool_sequence(messages: list[object]) -> list[str]:
    """create_agent 실행 결과 messages에서 Tool 호출 순서를 추출합니다."""
    used_tool_names = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
            else:
                tool_name = getattr(tool_call, "name", None)
            if tool_name:
                used_tool_names.append(tool_name)
    return used_tool_names


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

SYSTEM_PROMPT = (
    "너는 문서 기반 챗봇이다. 일반 대화도 가능하다. "
    "문서 검색이 필요하면 search_documents를 사용하고, 시간 질문은 get_current_time, "
    "컬렉션 상태는 get_collection_stats, 컬렉션 목록은 list_collections, "
    "chunk 상세는 get_chunk_detail을 사용해라. Tool이 필요 없으면 그냥 답변해라. "
    "사용자가 먼저 컬렉션 상태를 확인하라고 요청하면 get_collection_stats를 먼저 사용해라. "
    "embedding_count가 0보다 크면 그 다음 search_documents를 사용해 문서를 검색해라."
)

agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

QUESTION = (
    "먼저 현재 컬렉션 상태를 확인해줘. "
    "embedding_count가 0보다 크면, 그 다음 문서에서 생성형 AI 기술 동향을 검색해서 요약해줘."
)

print("=" * 80)
print("11. Multi-tool create_agent")
print("=" * 80)
print(f"질문: {QUESTION}")

result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
used_tool_names = get_used_tool_sequence(result["messages"])

print("\n[사용한 Tool 순서]")
print(" -> ".join(used_tool_names) if used_tool_names else "없음")

print("\n[최종 출력]")
print(result["messages"][-1].content)
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
TOOL_MAP = {tool_item.name: tool_item for tool_item in TOOLS}


from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI


QUESTION = (
    "먼저 현재 컬렉션 상태를 확인해줘. "
    "embedding_count가 0보다 크면, 그 다음 문서에서 생성형 AI 기술 동향을 검색해서 요약해줘."
)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

messages = [
    HumanMessage(
        content=(
            "너는 여러 Tool을 순서대로 사용할 수 있는 문서 기반 챗봇이다. "
            "사용자가 컬렉션 상태를 먼저 확인하라고 하면 get_collection_stats를 먼저 사용해라. "
            "문서가 있다고 판단되면 그 다음 search_documents를 사용해라. "
            "Tool 결과를 본 뒤 최종 답변을 만들어라.\n\n"
            f"질문: {QUESTION}"
        )
    )
]

print("=" * 80)
print("10. Multi-tool 수동 Tool Calling")
print("=" * 80)
print(f"질문: {QUESTION}")

response = llm_with_tools.invoke(messages)
messages.append(response)
step = 1
max_steps = 5

while response.tool_calls and step <= max_steps:
    print(f"\n[Tool Calling Step {step}]")
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"- Tool 요청: {tool_name} | args={tool_args}")
        selected_tool = TOOL_MAP[tool_name]
        tool_result = selected_tool.invoke(tool_args)
        print(f"- Tool 결과 preview: {tool_result[:500]}")
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    response = llm_with_tools.invoke(messages)
    messages.append(response)
    step += 1

print("\n[최종 답변]")
print(response.content)
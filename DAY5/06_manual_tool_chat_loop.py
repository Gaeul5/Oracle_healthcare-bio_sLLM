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


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

system_instruction = (
    "너는 문서 기반 챗봇이다. 일반 대화도 가능하다. "
    "문서 검색이 필요하면 search_documents를 사용하고, 시간 질문은 get_current_time, "
    "컬렉션 상태는 get_collection_stats, 컬렉션 목록은 list_collections, "
    "chunk 상세는 get_chunk_detail을 사용해라. Tool이 필요 없으면 그냥 답변해라."
)

print("=" * 80)
print("06. 수동 Tool Calling 채팅 루프")
print("종료하려면 quit 또는 exit 입력")
print("=" * 80)

while True:
    user_input = input("\n질문> ").strip()
    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    messages = [HumanMessage(content=f"{system_instruction}\n\n질문: {user_input}")]
    first_response = llm_with_tools.invoke(messages)
    messages.append(first_response)

    if not first_response.tool_calls:
        print("\n[Tool 호출 없음]")
        print(first_response.content)
        continue

    print("\n[Tool 호출 목록]")
    for tool_call in first_response.tool_calls:
        print(f"- {tool_call['name']} | args={tool_call['args']}")
        selected_tool = TOOL_MAP[tool_call["name"]]
        tool_result = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    second_response = llm_with_tools.invoke(messages)
    print("\n[최종 답변]")
    print(second_response.content)
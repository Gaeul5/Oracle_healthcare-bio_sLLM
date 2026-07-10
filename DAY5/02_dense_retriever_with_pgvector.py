from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

COLLECTION_NAME = "day5_pgvector_tool_demo"
QUERY = "생성형 AI의 기술 동향을 알려줘"
TOP_K = 3

embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

db_config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "testdb"),
    "user": os.getenv("POSTGRES_USER", "test"),
    "password": os.getenv("POSTGRES_PASSWORD", "5748"),
}

connection_string = PGVector.connection_string_from_db_params(driver="psycopg", **db_config)
embeddings_model = OpenAIEmbeddings(model=embedding_model_name)

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=COLLECTION_NAME,
    connection=connection_string,
    use_jsonb=True,
)

print("=" * 80)
print("02. PGVector VectorStore Retriever")
print("=" * 80)
print(f"질문: {QUERY}")

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
results = retriever.invoke(QUERY)

for idx, doc in enumerate(results, 1):
    print(f"\n[결과 {idx}]")
    print(f"내용: {doc.page_content[:200]}...")
    print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
    print(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
    print("-" * 80)
from __future__ import annotations

from common import DEFAULT_COLLECTION_NAME, get_connection_string, get_embeddings, print_documents
from langchain_postgres import PGVector

print("=" * 80)
print("03. 이미 저장된 LangChain PGVector collection 재사용")
print("- PDF를 다시 읽지 않습니다.")
print("- 같은 collection_name으로 VectorStore를 다시 연결합니다.")
print("=" * 80)

embeddings = get_embeddings()
connection_string = get_connection_string()

# 01번에서 저장한 collection_name을 그대로 사용합니다.
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=DEFAULT_COLLECTION_NAME,
    connection=connection_string,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "생성형 AI의 기술 동향 알려줘"
results = retriever.invoke(query)

print_documents(results, title=f"기존 collection 재사용 검색 결과: {query}")
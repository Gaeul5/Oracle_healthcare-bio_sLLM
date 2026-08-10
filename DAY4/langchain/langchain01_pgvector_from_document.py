from __future__ import annotations

from common import (
    DEFAULT_COLLECTION_NAME,
    get_connection_string,
    get_embeddings,
    load_and_split_pdf,
    print_documents,
)
from langchain_postgres import PGVector

print("=" * 80)
print("01. LangChain PGVector.from_documents + as_retriever")
print("- PDF 로드 → split → embedding → PostgreSQL(pgvector) 저장 → Retriever 검색")
print("=" * 80)

# 1. PDF 로드 및 chunk 분리
#    여기서는 LangChain 표준 흐름을 보여주기 위해 PDF를 직접 읽습니다.
documents = load_and_split_pdf()
print(f"분할된 chunk 개수: {len(documents)}")

# 2. Embedding 모델과 PostgreSQL 연결 준비
embeddings = get_embeddings()
connection_string = get_connection_string()

# 3. PGVector에 저장
#    실행하면 langchain_pg_collection, langchain_pg_embedding 테이블이 생성됩니다.
#    pre_delete_collection=True는 같은 collection_name의 기존 데이터를 지우고 다시 넣습니다.
vectorstore = PGVector.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=DEFAULT_COLLECTION_NAME,
    connection=connection_string,
    pre_delete_collection=True,
)

print(f"저장 완료 collection_name: {DEFAULT_COLLECTION_NAME}")
print("생성/사용되는 테이블: langchain_pg_collection, langchain_pg_embedding")

# 4. VectorStore를 Retriever로 변환
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 검색 실행
query = "생성형 AI의 기술 동향 알려줘"
results = retriever.invoke(query)

print_documents(results, title=f"Dense Retriever 결과: {query}")
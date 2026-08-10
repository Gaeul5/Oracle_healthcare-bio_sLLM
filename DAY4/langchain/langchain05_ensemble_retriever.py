from __future__ import annotations

from common import (
    ENSEMBLE_COLLECTION_NAME,
    get_connection_string,
    get_embeddings,
    load_and_split_pdf,
    print_documents,
)
from langchain_community.retrievers import BM25Retriever
from langchain_postgres import PGVector

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    # LangChain 버전에 따라 classic 패키지에 있을 수 있습니다.
    from langchain_classic.retrievers import EnsembleRetriever

print("=" * 80)
print("05. EnsembleRetriever Hybrid 검색")
print("- Dense Vector Retriever + Sparse BM25 Retriever")
print("=" * 80)

# 1. PDF 로드 및 chunk 분리
documents = load_and_split_pdf()
print(f"분할된 chunk 개수: {len(documents)}")

# 2. Dense Retriever 준비
embeddings = get_embeddings()
connection_string = get_connection_string()

vectorstore = PGVector.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=ENSEMBLE_COLLECTION_NAME,
    connection=connection_string,
    pre_delete_collection=True,
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Sparse Retriever 준비
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# 4. Ensemble Retriever 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

query = "생성형 AI 기술 동향"
results = ensemble_retriever.invoke(query)

print_documents(results, title=f"Ensemble Hybrid Retriever 결과: {query}")
from __future__ import annotations

from common import load_and_split_pdf, print_documents
from langchain_community.retrievers import BM25Retriever

print("=" * 80)
print("04. BM25Retriever Sparse 검색")
print("- PDF → split → Document 목록")
print("- BM25Retriever.from_documents(documents)")
print("- PostgreSQL에 저장하지 않고 Python 메모리에서 키워드 검색합니다.")
print("=" * 80)

documents = load_and_split_pdf()
print(f"분할된 chunk 개수: {len(documents)}")

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

query = "생성형 AI"
results = bm25_retriever.invoke(query)

print_documents(results, title=f"BM25 Sparse Retriever 결과: {query}")
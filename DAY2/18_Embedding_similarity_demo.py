from __future__ import annotations
import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
documents = [
"토마토 주스를 먹을 예정입니다.",
"생성형 AI 서비스가 기업 업무에 도입되고 있습니다.",
"오늘 점심 메뉴는 김치찌개입니다.",
]
query = "오늘 점심 메뉴는 무엇인가요?"
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODE", "text-embedding-3-small"))
doc_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)
scores = []
for doc, vec in zip(documents, doc_vectors):
    score = cosine_similarity(query_vector, vec)
    scores.append((score, doc))
scores.sort(reverse=True)
print("질문:", query)
print("--- 유사도 순위---")
for score, doc in scores:
    print(f"{score:.4f} | {doc}")
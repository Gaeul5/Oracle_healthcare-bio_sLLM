"""Streamlit RAG 실습 준비 상태 확인 파일."""

from __future__ import annotations

import importlib.util
from common import env, get_db_counts


def check_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


print("Day 3 추가 실습: Streamlit 준비 상태 확인\n")

print("[1] Python 패키지 확인")
for package in ["streamlit", "pandas", "psycopg", "langchain_openai"]:
    ok = check_package(package)
    print(f"- {package}: {'OK' if ok else 'NOT FOUND'}")

print("\n[2] .env 확인")
api_key = env("OPENAI_API_KEY")
print(f"- OPENAI_API_KEY: {'입력됨' if api_key else '비어 있음'}")
print(f"- OPENAI_MODEL: {env('OPENAI_MODEL', 'gpt-4o-mini')}")
print(f"- OPENAI_EMBEDDING_MODEL: {env('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}")
print(f"- POSTGRES_HOST: {env('POSTGRES_HOST', 'localhost')}")
print(f"- POSTGRES_DB: {env('POSTGRES_DB')}")

print("\n[3] Day 3 DB 데이터 확인")
try:
    counts = get_db_counts()
    for table, count in counts.items():
        print(f"- {table}: {count}")

    if counts.get("rag_chunks", 0) == 0 or counts.get("rag_embeddings", 0) == 0:
        print("\n주의: rag_chunks 또는 rag_embeddings가 비어 있습니다.")
        print("Day 3의 PDF/TXT embedding 저장 실습을 먼저 완료하세요.")
    else:
        print("\nStreamlit RAG 실습 준비가 완료되었습니다.")
except Exception as e:
    print("\nDB 확인 중 오류가 발생했습니다.")
    print(e)
    print(".env의 DB 접속 정보와 PostgreSQL 실행 상태를 확인하세요.")

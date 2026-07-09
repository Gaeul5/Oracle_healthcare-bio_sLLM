from __future__ import annotations

import os

from common import connect_db, get_connection_string, get_embedding_model_name

REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]

print("=" * 80)
print("00. 환경변수와 DB 연결 확인")
print("=" * 80)

missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
if missing:
    print("누락된 환경변수:")
    for key in missing:
        print(f"- {key}")
else:
    print("필수 환경변수가 준비되었습니다.")

print(f"Embedding model: {get_embedding_model_name()}")
print(f"Connection string preview: {get_connection_string().replace(os.getenv('POSTGRES_PASSWORD', ''), '****')}")

try:
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

    print("DB 연결 성공")
    print(version)
    print("pgvector extension 확인 완료")
except Exception as exc:
    print("DB 연결 실패")
    print(type(exc).__name__, exc)
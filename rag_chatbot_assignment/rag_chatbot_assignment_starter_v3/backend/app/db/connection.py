from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import psycopg

from backend.app.core.config import settings


def get_connection() -> psycopg.Connection:
    """PostgreSQL 연결을 생성합니다.

    이 함수는 완성 코드로 제공합니다.
    학생들은 이 connection을 사용해서 SQL을 작성하면 됩니다.
    """
    return psycopg.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


@contextmanager
def db_session() -> Generator[psycopg.Connection, None, None]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_database_connection() -> tuple[bool, str]:
    """DB 연결과 기존 실습 테이블 존재 여부를 확인합니다."""
    missing_env = settings.missing_required_env()
    if missing_env:
        return False, "누락된 환경변수: " + ", ".join(missing_env)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        to_regclass('public.rag_documents') AS rag_documents,
                        to_regclass('public.rag_chunks') AS rag_chunks,
                        to_regclass('public.rag_embeddings') AS rag_embeddings
                    """
                )
                row = cur.fetchone()

        table_names = ["rag_documents", "rag_chunks", "rag_embeddings"]
        missing_tables = [name for name, value in zip(table_names, row or []) if value is None]
        if missing_tables:
            return False, "없는 테이블: " + ", ".join(missing_tables)

        return True, "PostgreSQL 연결 및 기존 RAG 테이블 확인 완료"
    except Exception as exc:
        return False, f"DB 연결 실패: {exc}"


def to_pgvector(vector: list[float]) -> str:
    """Python list[float]를 pgvector literal 문자열로 바꿉니다.

    예: [0.1, 0.2, 0.3] -> '[0.1,0.2,0.3]'
    """
    return "[" + ",".join(str(float(value)) for value in vector) + "]"

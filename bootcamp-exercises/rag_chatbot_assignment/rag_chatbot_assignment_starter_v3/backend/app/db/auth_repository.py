from __future__ import annotations

from psycopg.rows import dict_row

from backend.app.db.connection import db_session


def create_user(email: str, password_hash: str, name: str) -> dict:
    """users 테이블에 새 사용자를 저장하고 저장된 row를 반환합니다."""
    sql = """
        INSERT INTO users (email, password_hash, name)
        VALUES (%s, %s, %s)
        RETURNING id, email, name, created_at;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (email, password_hash, name))
            return cur.fetchone()


def find_user_by_email(email: str) -> dict | None:
    """email로 사용자를 조회합니다. password_hash 검증을 위해 password_hash도 함께 반환합니다."""
    sql = """
        SELECT id, email, password_hash, name, created_at
        FROM users
        WHERE email = %s;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (email,))
            return cur.fetchone()


def find_user_by_id(user_id: int) -> dict | None:
    """user_id로 사용자를 조회합니다. (password_hash는 반환하지 않습니다)"""
    sql = """
        SELECT id, email, name, created_at
        FROM users
        WHERE id = %s;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (user_id,))
            return cur.fetchone()

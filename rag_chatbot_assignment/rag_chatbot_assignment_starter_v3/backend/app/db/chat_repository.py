from __future__ import annotations

import json

from psycopg.rows import dict_row

from backend.app.db.connection import db_session


def create_session(user_id: int, title: str) -> dict:
    sql = """
        INSERT INTO chat_sessions (user_id, title)
        VALUES (%s, %s)
        RETURNING id AS session_id, title, created_at;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (user_id, title))
            return cur.fetchone()


def find_sessions_by_user(user_id: int) -> list[dict]:
    sql = """
        SELECT id AS session_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE user_id = %s AND deleted_at IS NULL
        ORDER BY updated_at DESC;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (user_id,))
            return cur.fetchall()


def find_session_by_id(session_id: int) -> dict | None:
    """소유자 확인용으로 user_id도 함께 반환합니다."""
    sql = """
        SELECT id AS session_id, user_id, title, created_at, updated_at
        FROM chat_sessions
        WHERE id = %s AND deleted_at IS NULL;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (session_id,))
            return cur.fetchone()


def update_session_title(session_id: int, title: str) -> dict:
    sql = """
        UPDATE chat_sessions
        SET title = %s, updated_at = NOW()
        WHERE id = %s
        RETURNING id AS session_id, title;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (title, session_id))
            return cur.fetchone()


def soft_delete_session(session_id: int) -> None:
    sql = """
        UPDATE chat_sessions
        SET deleted_at = NOW()
        WHERE id = %s;
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))


def find_messages_by_session(session_id: int) -> list[dict]:
    sql = """
        SELECT id AS message_id, role, content, metadata, created_at
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at ASC;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (session_id,))
            return cur.fetchall()


def insert_message(
    session_id: int,
    user_id: int,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> dict:
    """user/assistant 메시지를 chat_messages에 저장합니다."""
    sql = """
        INSERT INTO chat_messages (session_id, user_id, role, content, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        RETURNING id AS message_id, created_at;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (session_id, user_id, role, content, json.dumps(metadata or {})))
            return cur.fetchone()


def touch_session_updated_at(session_id: int) -> None:
    """새 메시지가 생길 때마다 채팅방의 updated_at을 갱신합니다.

    이래야 '최근 대화가 있는 채팅방이 위로' 정렬이 실제로 동작합니다.
    """
    sql = """
        UPDATE chat_sessions
        SET updated_at = NOW()
        WHERE id = %s;
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))


def insert_tool_call(
    session_id: int,
    user_id: int,
    assistant_message_id: int,
    tool_name: str,
    tool_input: dict,
    tool_output_preview: str | None,
    success: bool,
    error_message: str | None,
) -> None:
    """Tool 호출 1건을 chat_tool_calls에 기록합니다."""
    sql = """
        INSERT INTO chat_tool_calls (
            session_id, user_id, assistant_message_id,
            tool_name, tool_input, tool_output_preview, success, error_message
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s);
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    session_id,
                    user_id,
                    assistant_message_id,
                    tool_name,
                    json.dumps(tool_input, ensure_ascii=False, default=str),
                    tool_output_preview,
                    success,
                    error_message,
                ),
            )

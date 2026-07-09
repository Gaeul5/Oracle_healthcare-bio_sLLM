from __future__ import annotations

from psycopg.rows import dict_row

from backend.app.db.connection import db_session, to_pgvector


# ============================================================
# TODO 1. 문서 목록 조회
# ============================================================
def find_all_documents() -> list[dict]:
    """rag_documents 기준으로 문서 목록을 조회합니다.

    학생 구현 목표:
    - rag_documents를 조회한다.
    - rag_chunks를 LEFT JOIN해서 chunk_count를 계산한다.
    - chunk_count가 0보다 크면 status를 indexed로 표시한다.

    힌트 SQL:
        SELECT
            d.id,
            d.title,
            d.file_name,
            d.file_type,
            d.page_count,
            d.created_at,
            COUNT(c.id) AS chunk_count
        FROM rag_documents d
        LEFT JOIN rag_chunks c ON d.id = c.document_id
        GROUP BY d.id
        ORDER BY d.id DESC;
    """
    sql = """
        SELECT
            d.id,
            d.title,
            d.file_name,
            d.file_type,
            d.page_count,
            d.created_at,
            COUNT(c.id) AS chunk_count
        FROM rag_documents d
        LEFT JOIN rag_chunks c ON d.id = c.document_id
        GROUP BY d.id
        ORDER BY d.id DESC;
    """
    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    for row in rows:
        row["status"] = "indexed" if row["chunk_count"] > 0 else "pending"

    return rows


# ============================================================
# TODO 2. 문서 정보 저장
# ============================================================
def insert_document(title: str, file_name: str, file_type: str, page_count: int | None) -> int:
    """rag_documents에 문서 정보를 저장하고 document_id를 반환합니다."""
    sql = """
        INSERT INTO rag_documents (title, file_name, file_type, page_count)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (title, file_name, file_type, page_count))
            document_id = cur.fetchone()[0]

    return document_id


# ============================================================
# TODO 3. chunk 저장
# ============================================================
def insert_chunk(document_id: int, chunk_index: int, page_number: int | None, content: str) -> int:
    """rag_chunks에 chunk 본문을 저장하고 chunk_id를 반환합니다."""
    sql = """
        INSERT INTO rag_chunks (document_id, chunk_index, page_number, content)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (document_id, chunk_index, page_number, content))
            chunk_id = cur.fetchone()[0]

    return chunk_id


# ============================================================
# TODO 4. embedding 저장
# ============================================================
def insert_embedding(chunk_id: int, embedding_model: str, embedding: list[float]) -> int:
    """rag_embeddings에 embedding vector를 저장합니다.

    힌트:
    - to_pgvector(embedding)을 사용하면 list[float]를 '[...]' 문자열로 바꿀 수 있습니다.
    - SQL에서는 %s::vector 형태로 넣으면 됩니다.
    """
    sql = """
        INSERT INTO rag_embeddings (chunk_id, embedding_model, embedding)
        VALUES (%s, %s, %s::vector)
        RETURNING id;
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (chunk_id, embedding_model, to_pgvector(embedding)))
            embedding_id = cur.fetchone()[0]

    return embedding_id


# ============================================================
# TODO 5. vector similarity search
# ============================================================
def search_similar_chunks(query_embedding: list[float], embedding_model: str, top_k: int) -> list[dict]:
    """질문 embedding과 유사한 chunk를 검색합니다.

    학생 구현 목표:
    - rag_embeddings.embedding과 query_embedding 사이의 거리를 계산한다.
    - rag_chunks, rag_documents와 JOIN한다.
    - distance가 낮은 순서로 top_k개를 반환한다.

    힌트 SQL:
        SELECT
            d.id AS document_id,
            d.title AS document_title,
            d.file_name,
            d.file_type,
            c.id AS chunk_id,
            c.chunk_index,
            c.page_number,
            c.content,
            e.embedding <=> %s::vector AS distance
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        JOIN rag_documents d ON c.document_id = d.id
        WHERE e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """
    sql = """
        SELECT
            d.id AS document_id,
            d.title AS document_title,
            d.file_name,
            d.file_type,
            c.id AS chunk_id,
            c.chunk_index,
            c.page_number,
            c.content,
            e.embedding <=> %s::vector AS distance
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        JOIN rag_documents d ON c.document_id = d.id
        WHERE e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """
    query_vector = to_pgvector(query_embedding)

    with db_session() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (query_vector, embedding_model, query_vector, top_k))
            rows = cur.fetchall()

    return rows


# ============================================================
# TODO 6. 문서 삭제
# ============================================================
def delete_document(document_id: int) -> None:
    """rag_documents에서 문서를 삭제합니다.

    기존 schema에 ON DELETE CASCADE가 있으므로 rag_chunks, rag_embeddings도 함께 삭제됩니다.
    """
    sql = """
        DELETE FROM rag_documents
        WHERE id = %s;
    """
    with db_session() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (document_id,))
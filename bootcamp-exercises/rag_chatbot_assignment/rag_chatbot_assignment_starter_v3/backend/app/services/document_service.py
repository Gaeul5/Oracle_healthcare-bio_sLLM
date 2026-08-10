from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile

from backend.app.core.config import settings
from backend.app.utils.filename import get_file_type, make_safe_filename
from backend.app.utils.file_loader import validate_file_extension


def save_upload_file(upload_file: UploadFile) -> Path:
    """업로드된 파일을 backend/uploads 폴더에 저장합니다.

    이 함수는 과제 핵심이 아니므로 완성 코드로 제공합니다.
    """
    validate_file_extension(upload_file.filename or "")
    safe_name = make_safe_filename(upload_file.filename or "uploaded_file")
    save_path = settings.UPLOAD_DIR / safe_name

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return save_path


def split_documents(documents):
    """LangChain Document 목록을 chunk로 나눕니다.

    TODO:
    - RecursiveCharacterTextSplitter를 생성합니다.
    - chunk_size는 800~1000 정도로 설정합니다.
    - chunk_overlap은 100~150 정도로 설정합니다.
    - splitter.split_documents(documents)를 반환합니다.

    힌트:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
    )
    return splitter.split_documents(documents)


def index_document(file_path: Path, user_id: int, title: str | None = None) -> dict:
    """파일을 읽고 chunk/embedding을 생성해 기존 PostgreSQL 테이블에 저장합니다.

    TODO:
    1. load_file_as_documents(file_path)로 문서를 읽습니다.
    2. split_documents()로 chunk를 만듭니다.
    3. rag_documents에 문서 정보를 저장합니다 (user_id 포함).
    4. 각 chunk를 rag_chunks에 저장합니다.
    5. 각 chunk 본문을 embedding합니다.
    6. rag_embeddings에 vector를 저장합니다.
    7. 문서 ID, 제목, chunk_count를 dict로 반환합니다.
    """
    from backend.app.utils.file_loader import load_file_as_documents
    from backend.app.services.embedding_service import embed_text
    from backend.app.db.repository import (
        insert_document,
        insert_chunk,
        insert_embedding,
    )

    title = title or file_path.stem
    documents = load_file_as_documents(file_path)
    chunks = split_documents(documents)

    document_id = insert_document(
        user_id=user_id,
        title=title,
        file_name=file_path.name,
        file_type=get_file_type(file_path),
        page_count=len(documents),
    )

    for chunk_index, chunk in enumerate(chunks):
        chunk_id = insert_chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            page_number=chunk.metadata.get("page"),
            content=chunk.page_content,
        )
        embedding = embed_text(chunk.page_content)
        insert_embedding(
            chunk_id=chunk_id,
            embedding_model=settings.OPENAI_EMBEDDING_MODEL,
            embedding=embedding,
        )

    return {
        "document_id": document_id,
        "title": title,
        "chunk_count": len(chunks),
    }


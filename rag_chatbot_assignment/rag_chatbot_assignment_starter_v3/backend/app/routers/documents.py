from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.db import repository
from backend.app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def get_documents():
    """문서 목록을 조회합니다.

    TODO:
    1. backend.app.db.repository.find_all_documents()를 호출합니다.
    2. {"documents": documents} 형태로 반환합니다.
    """
    documents = repository.find_all_documents()

    return {"documents": documents}


@router.post("")
def upload_document(title: str = Form(""), file: UploadFile = File(...)):
    """문서를 업로드하고 indexing합니다.

    TODO:
    1. document_service.save_upload_file(file)로 파일을 저장합니다.
    2. document_service.index_document(saved_path, title)로 RAG 저장을 수행합니다.
    3. 결과 dict를 반환합니다.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 비어 있습니다.")

    saved_path = document_service.save_upload_file(file)
    result = document_service.index_document(saved_path, title)

    return {
        "message": "uploaded",
        "received_title": title,
        "received_file_name": file.filename,
        "result": result,
    }


@router.delete("/{document_id}")
def delete_document(document_id: int):
    """문서를 삭제합니다.

    TODO:
    1. repository.delete_document(document_id)를 호출합니다.
    2. uploads 폴더에 저장된 원본 파일도 삭제할지 결정합니다.
    3. 삭제 결과를 반환합니다.
    """
    repository.delete_document(document_id)

    return {"message": "deleted", "document_id": document_id}

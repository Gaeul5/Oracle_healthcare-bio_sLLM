from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.app.core.deps import get_current_user_id
from backend.app.db import repository
from backend.app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def get_documents(current_user_id: int = Depends(get_current_user_id)):
    """현재 로그인한 사용자의 문서 목록을 조회합니다."""
    documents = repository.find_all_documents(current_user_id)

    return {"documents": documents}


@router.post("")
def upload_document(
    title: str = Form(""),
    file: UploadFile = File(...),
    current_user_id: int = Depends(get_current_user_id),
):
    """문서를 업로드하고 indexing합니다. 업로드한 사람의 user_id로 저장됩니다."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 비어 있습니다.")

    saved_path = document_service.save_upload_file(file)
    result = document_service.index_document(saved_path, user_id=current_user_id, title=title)

    return {
        "message": "uploaded",
        "received_title": title,
        "received_file_name": file.filename,
        "result": result,
    }


@router.delete("/{document_id}")
def delete_document(document_id: int, current_user_id: int = Depends(get_current_user_id)):
    """현재 로그인한 사용자가 소유한 문서만 삭제합니다.

    TODO:
    2. uploads 폴더에 저장된 원본 파일도 삭제할지 결정합니다.
    """
    deleted = repository.delete_document(document_id, current_user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    return {"message": "deleted", "document_id": document_id}

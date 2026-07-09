from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    top_k: int = 4


@router.post("")
def chat(request: ChatRequest):
    """사용자 질문에 대해 RAG 답변을 생성합니다.

    TODO:
    1. request.message가 비어 있는지 검사합니다.
    2. backend.app.services.rag_service.ask()를 호출합니다.
    3. {"answer": answer, "sources": sources} 형태로 반환합니다.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    return rag_service.ask(question=request.message, top_k=request.top_k)

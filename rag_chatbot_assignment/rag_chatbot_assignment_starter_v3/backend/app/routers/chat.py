from __future__ import annotations

import fastapi
from pydantic import BaseModel

from backend.app.core.deps import get_current_user_id
from backend.app.db import chat_repository
from backend.app.services import chat_service

router = fastapi.APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: int
    message: str
    mode: str = "manual"
    explanation_level: str = "friendly"


class CreateSessionRequest(BaseModel):
    title: str = ""


class UpdateSessionRequest(BaseModel):
    title: str


def _get_owned_session(session_id: int, current_user_id: int) -> dict:
    """session_id가 current_user_id 소유인지 확인합니다. 아니면 404를 던집니다.

    다른 사용자의 session_id를 알아내도 존재 여부를 알 수 없도록
    권한 없음(403)이 아니라 없음(404)으로 응답합니다.
    """
    session = chat_repository.find_session_by_id(session_id)
    if not session or session["user_id"] != current_user_id:
        raise fastapi.HTTPException(status_code=404, detail="채팅방을 찾을 수 없습니다.")
    return session


@router.post("")
def chat(
    request: ChatRequest,
    current_user_id: int = fastapi.Depends(get_current_user_id),
):
    """채팅방에 메시지를 보내고 답변을 받습니다.

    mode="manual"이면 수동 tool_calls 반복 처리, mode="agent"면 AgentExecutor로 처리합니다.
    둘 다 응답 구조(answer/tools/sources)는 동일합니다.
    """
    if not request.message.strip():
        raise fastapi.HTTPException(status_code=400, detail="질문을 입력해주세요.")

    _get_owned_session(request.session_id, current_user_id)

    run_chat = (
        chat_service.run_tool_chat_agent
        if request.mode == "agent"
        else chat_service.run_tool_chat_manual
    )

    return run_chat(
        user_id=current_user_id,
        session_id=request.session_id,
        message=request.message,
        explanation_level=request.explanation_level,
    )


@router.post("/sessions")
def create_session(
    request: CreateSessionRequest,
    current_user_id: int = fastapi.Depends(get_current_user_id),
):
    title = request.title.strip() or "새 채팅"
    return chat_repository.create_session(user_id=current_user_id, title=title)


@router.get("/sessions")
def list_sessions(current_user_id: int = fastapi.Depends(get_current_user_id)):
    sessions = chat_repository.find_sessions_by_user(current_user_id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: int,
    current_user_id: int = fastapi.Depends(get_current_user_id),
):
    _get_owned_session(session_id, current_user_id)

    rows = chat_repository.find_messages_by_session(session_id)
    messages = []
    for row in rows:
        metadata = row["metadata"] or {}
        message = {
            "message_id": row["message_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        if row["role"] == "assistant":
            message["tools"] = metadata.get("tools", [])
            message["sources"] = metadata.get("sources", [])
        messages.append(message)

    return {"session_id": session_id, "messages": messages}


@router.patch("/sessions/{session_id}")
def rename_session(
    session_id: int,
    request: UpdateSessionRequest,
    current_user_id: int = fastapi.Depends(get_current_user_id),
):
    _get_owned_session(session_id, current_user_id)

    title = request.title.strip()
    if not title:
        raise fastapi.HTTPException(status_code=400, detail="제목을 입력해주세요.")

    return chat_repository.update_session_title(session_id, title)


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    current_user_id: int = fastapi.Depends(get_current_user_id),
):
    _get_owned_session(session_id, current_user_id)

    chat_repository.soft_delete_session(session_id)
    return {"message": "deleted", "session_id": session_id}

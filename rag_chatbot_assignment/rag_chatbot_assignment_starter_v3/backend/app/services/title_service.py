from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.app.core.config import settings

TITLE_SYSTEM_PROMPT = (
    "다음은 채팅방의 첫 사용자 메시지다. 이 채팅방에 어울리는 제목을 "
    "한국어 명사구로 15자 이내로 만들어라. 따옴표나 마침표 없이 제목만 출력해라."
)

DEFAULT_TITLE = "새 채팅"
MAX_TITLE_LENGTH = 15


def generate_chat_title(first_user_message: str) -> str:
    """첫 사용자 메시지를 보고 짧은 채팅방 제목을 생성합니다."""
    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content=TITLE_SYSTEM_PROMPT),
            HumanMessage(content=first_user_message),
        ]
    )

    title = response.content.strip().strip('"').strip("'").strip()
    if len(title) > MAX_TITLE_LENGTH:
        title = title[:MAX_TITLE_LENGTH]

    return title or DEFAULT_TITLE

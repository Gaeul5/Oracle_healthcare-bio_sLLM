from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="메시지 역할: user 또는 assistant")
    content: str = Field(description="메시지 내용")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="사용자 입력")
    history: list[ChatMessage] = Field(default_factory=list, description="이전 대화 기록")
    style: str = Field(default="friendly", description="답변 스타일")


class ChatbotAnswer(BaseModel):
    answer: str = Field(description="챗봇의 최종 답변")
    key_points: list[str] = Field(description="답변의 핵심 포인트")
    next_question: str = Field(description="사용자에게 이어서 물어볼 만한 질문")


class ChatResponse(BaseModel):
    answer: str


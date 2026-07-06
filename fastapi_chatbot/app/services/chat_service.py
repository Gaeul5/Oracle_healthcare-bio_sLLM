import os
from collections.abc import Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.schemas import ChatMessage, ChatbotAnswer


load_dotenv()


STYLE_INSTRUCTIONS = {
    "friendly": "친절하고 자연스러운 한국어로 답변하세요.",
    "easy": "처음 배우는 사람도 이해할 수 있게 쉬운 단어와 짧은 문장으로 답변하세요.",
    "expert": "핵심 개념, 실무 관점, 주의점을 포함해 전문적으로 답변하세요.",
}


class ChatService:
    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.5,
        )
        self.parser = PydanticOutputParser(pydantic_object=ChatbotAnswer)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 Python, FastAPI, LangChain을 도와주는 한국어 학습 챗봇입니다.\n"
                    "사용자의 질문에 정확하고 실용적으로 답변하세요.\n\n"
                    "[답변 스타일]\n{style_instruction}\n\n"
                    "[구조화 출력 지시]\n{format_instructions}",
                ),
                MessagesPlaceholder("history"),
                ("human", "{message}"),
            ]
        )

        self.stream_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 Python, FastAPI, LangChain을 도와주는 한국어 학습 챗봇입니다.\n"
                    "사용자의 질문에 정확하고 실용적으로 답변하세요.\n\n"
                    "[답변 스타일]\n{style_instruction}",
                ),
                MessagesPlaceholder("history"),
                ("human", "{message}"),
            ]
        )

        # LCEL: prompt | model | parser
        self.structured_chain = self.prompt | self.model | self.parser

        # Runnable.stream() 데모용 체인: 화면에 토큰을 실시간 출력한다.
        self.stream_chain = self.stream_prompt | self.model | StrOutputParser()

    def _to_langchain_messages(
        self, history: list[ChatMessage]
    ) -> list[HumanMessage | AIMessage]:
        messages: list[HumanMessage | AIMessage] = []

        for message in history:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                messages.append(AIMessage(content=message.content))

        return messages

    def _chain_input(
        self, message: str, history: list[ChatMessage], style: str
    ) -> dict[str, object]:
        return {
            "message": message,
            "history": self._to_langchain_messages(history),
            "style_instruction": STYLE_INSTRUCTIONS.get(
                style, STYLE_INSTRUCTIONS["friendly"]
            ),
            "format_instructions": self.parser.get_format_instructions(),
        }

    def invoke(self, message: str, history: list[ChatMessage], style: str) -> ChatbotAnswer:
        return self.structured_chain.invoke(self._chain_input(message, history, style))

    def stream(
        self, message: str, history: list[ChatMessage], style: str
    ) -> Generator[str, None, None]:
        chain_input = {
            "message": message,
            "history": self._to_langchain_messages(history),
            "style_instruction": STYLE_INSTRUCTIONS.get(
                style, STYLE_INSTRUCTIONS["friendly"]
            ),
        }

        # Runnable.stream() 사용
        for chunk in self.stream_chain.stream(chain_input):
            yield chunk


chat_service = ChatService()


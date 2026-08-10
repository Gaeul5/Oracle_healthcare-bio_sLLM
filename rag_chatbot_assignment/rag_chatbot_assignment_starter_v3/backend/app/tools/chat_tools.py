from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.app.db import chat_repository, repository
from backend.app.services import rag_service


class SearchDocumentsArgs(BaseModel):
    query: str = Field(description="검색할 질문 또는 키워드")
    top_k: int = Field(default=4, description="가져올 chunk 개수")
    document_id: int | None = Field(
        default=None, description="특정 문서 안에서만 검색하고 싶을 때 그 문서의 id. 모르면 비워둔다."
    )


class GetChunkDetailArgs(BaseModel):
    chunk_id: int = Field(description="원문을 확인할 chunk의 id")


def build_tools_for_user(user_id: int, session_id: int) -> list[StructuredTool]:
    """user_id/session_id를 클로저로 고정한 Tool 목록을 만듭니다.

    LLM은 이 두 값을 절대 파라미터로 넘기지 않습니다. 서버가 로그인 정보로
    이미 알고 있는 값을 요청마다 여기서 주입합니다.
    """

    def search_documents(query: str, top_k: int = 4, document_id: int | None = None) -> dict:
        if document_id is not None:
            document = repository.find_document_by_id(document_id)
            if not document or document["user_id"] != user_id:
                return {
                    "query": query,
                    "documents": [],
                    "sources": [],
                    "error": "등록된 문서가 없거나 접근 권한이 없습니다.",
                }

        rows = rag_service.search_similar_chunks(
            query, user_id=user_id, top_k=top_k, document_id=document_id
        )
        if not rows:
            return {
                "query": query,
                "documents": [],
                "sources": [],
                "message": "관련된 문서를 찾을 수 없습니다.",
            }

        documents = [
            {"chunk_id": row["chunk_id"], "content": row["content"], "score": round(row["distance"], 4)}
            for row in rows
        ]
        return {"query": query, "documents": documents, "sources": rag_service.build_sources(rows)}

    def get_document_stats() -> dict:
        return repository.get_document_stats(user_id)

    def get_current_time() -> dict:
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        return {"timezone": "Asia/Seoul", "current_time": now.strftime("%Y-%m-%d %H:%M:%S")}

    def list_documents() -> dict:
        rows = repository.find_all_documents(user_id)
        documents = [
            {
                "document_id": row["id"],
                "title": row["title"],
                "file_name": row["file_name"],
                "file_type": row["file_type"],
                "page_count": row["page_count"],
                "chunk_count": row["chunk_count"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
        return {"documents": documents}

    def get_chunk_detail(chunk_id: int) -> dict:
        chunk = repository.find_chunk_detail(chunk_id)
        if not chunk or chunk["user_id"] != user_id:
            return {"error": "chunk를 찾을 수 없거나 접근 권한이 없습니다."}

        return {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "file_name": chunk["file_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
        }

    def summarize_chat_history() -> dict:
        rows = chat_repository.find_messages_by_session(session_id)
        rows = [row for row in rows if row["role"] in ("user", "assistant")]
        if not rows:
            return {"summary": "아직 이 채팅방에는 대화 기록이 없습니다."}

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from backend.app.core.config import settings

        transcript = "\n".join(f"{row['role']}: {row['content']}" for row in rows[-10:])
        llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
        response = llm.invoke(
            [
                SystemMessage(content="아래 대화 내용을 한국어 3문장 이내로 요약해라."),
                HumanMessage(content=transcript),
            ]
        )
        return {"summary": response.content}

    return [
        StructuredTool.from_function(
            func=search_documents,
            name="search_documents",
            description=(
                "등록된 문서에서 질문과 관련된 내용을 의미 기반으로 검색한다. "
                "'등록된 문서', '업로드한 자료', '문서 기준', '자료에 따르면' 같은 표현이 있거나 "
                "문서 내용에 대해 물어볼 때 사용한다."
            ),
            args_schema=SearchDocumentsArgs,
        ),
        StructuredTool.from_function(
            func=get_document_stats,
            name="get_document_stats",
            description="현재 로그인한 사용자의 문서 개수, chunk 개수, embedding 개수를 조회한다.",
        ),
        StructuredTool.from_function(
            func=get_current_time,
            name="get_current_time",
            description="현재 시간이나 날짜를 물어볼 때 사용한다.",
        ),
        StructuredTool.from_function(
            func=list_documents,
            name="list_documents",
            description="현재 로그인한 사용자가 등록한 문서 목록을 조회한다.",
        ),
        StructuredTool.from_function(
            func=get_chunk_detail,
            name="get_chunk_detail",
            description="특정 chunk_id의 원문 내용을 조회한다. 출처를 클릭했거나 특정 chunk 원문을 요청할 때 사용한다.",
            args_schema=GetChunkDetailArgs,
        ),
        StructuredTool.from_function(
            func=summarize_chat_history,
            name="summarize_chat_history",
            description="현재 채팅방의 최근 대화 내용을 요약한다.",
        ),
    ]

from __future__ import annotations

import json

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from backend.app.core.config import settings
from backend.app.db import chat_repository
from backend.app.services import title_service
from backend.app.tools.chat_tools import build_tools_for_user

TOOL_OUTPUT_PREVIEW_LENGTH = 800

RECENT_MESSAGE_LIMIT = 10
MAX_TOOL_ITERATIONS = 3

SYSTEM_PROMPT = """너는 일반 대화와 문서 기반 질문을 모두 처리할 수 있는 LangChain 챗봇이다.

규칙:
1. 일반적인 인사, 잡담, 간단한 개념 설명은 Tool을 쓰지 않고 바로 답변한다.
2. 사용자가 "등록된 문서", "업로드한 문서", "문서 기준", "자료에 따르면" 같은 표현을 쓰면 search_documents Tool을 사용한다.
3. 현재 시각/날짜를 물으면 get_current_time Tool을 사용한다.
4. 저장된 문서/청크/임베딩 개수 등 DB 상태를 물으면 get_document_stats Tool을 사용한다.
5. 등록된 문서 목록을 물으면 list_documents Tool을 사용한다.
6. 특정 chunk_id 원문이나 출처 상세 내용을 물으면 get_chunk_detail Tool을 사용한다.
7. 지금까지의 대화 요약을 요청하면 summarize_chat_history Tool을 사용한다.
8. 사용자가 여러 작업을 순서대로 요청하면 필요한 Tool을 순서대로 사용할 수 있다.
9. Tool 결과에 sources가 있으면 답변에서 그 내용을 근거로 사용한다.
10. 출처는 직접 지어내지 않는다. Tool이 준 정보만 사실로 취급한다.
11. 다른 사용자의 문서나 채팅방에 접근하려는 요청은 실행할 수 없다고 답한다.
12. Tool이 필요 없는 질문에는 절대 Tool을 호출하지 않는다.

한국어로 답변해라."""

EXPLANATION_LEVELS = {
    "simple": "답변은 최대한 간단하게, 핵심만 짧게 요약해서 설명해라. 배경 설명이나 부연 설명은 생략해라.",
    "friendly": "답변은 친절하고 이해하기 쉽게, 필요하면 쉬운 예시를 들어가며 설명해라.",
    "professional": "답변은 전문 용어를 정확히 사용해서 상세하고 전문적으로 설명해라. 같은 분야 전문가에게 설명하듯 답변해라.",
}
DEFAULT_EXPLANATION_LEVEL = "friendly"


def _build_system_prompt(explanation_level: str) -> str:
    instruction = EXPLANATION_LEVELS.get(explanation_level, EXPLANATION_LEVELS[DEFAULT_EXPLANATION_LEVEL])
    return f"{SYSTEM_PROMPT}\n\n[설명 강도 지시사항]\n{instruction}"


def _load_recent_history(session_id: int) -> list:
    """session_id의 최근 대화를 LangChain 메시지 목록으로 변환합니다.

    Tool 결과는 metadata에만 남기고 여기서는 user/assistant 텍스트만
    최근 RECENT_MESSAGE_LIMIT개 사용합니다.
    """
    rows = chat_repository.find_messages_by_session(session_id)
    rows = [row for row in rows if row["role"] in ("user", "assistant")]
    rows = rows[-RECENT_MESSAGE_LIMIT:]

    history = []
    for row in rows:
        if row["role"] == "user":
            history.append(HumanMessage(content=row["content"]))
        else:
            history.append(AIMessage(content=row["content"]))
    return history


def _maybe_generate_title(session_id: int, history: list, first_message: str) -> None:
    """채팅방의 첫 사용자 메시지라면 제목을 자동 생성합니다.

    history가 비어있다는 건 이 채팅방에 아직 user/assistant 메시지가 하나도 없었다는 뜻입니다.
    이미 사용자가 제목을 직접 바꿔둔 경우(기본값이 아닌 경우)는 덮어쓰지 않습니다.
    """
    if history:
        return

    session = chat_repository.find_session_by_id(session_id)
    if not session or session["title"] != title_service.DEFAULT_TITLE:
        return

    title = title_service.generate_chat_title(first_message)
    chat_repository.update_session_title(session_id, title)


def _finalize_turn(
    session_id: int,
    user_id: int,
    answer: str,
    used_tools: list[dict],
    collected_sources: list[dict],
) -> dict:
    """assistant 답변을 저장하고, Tool 호출 기록을 남기고, 공통 응답 형태를 만듭니다.

    manual/agent 두 방식 모두 이 함수로 끝나야 응답 구조가 완전히 동일해집니다.
    """
    # tool_call insert에만 필요한 raw_output은 API 응답/메시지 metadata에는 남기지 않습니다.
    raw_outputs = [record.pop("raw_output", None) for record in used_tools]

    saved = chat_repository.insert_message(
        session_id,
        user_id,
        "assistant",
        answer,
        metadata={"tools": used_tools, "sources": collected_sources},
    )
    chat_repository.touch_session_updated_at(session_id)

    for record, raw_output in zip(used_tools, raw_outputs):
        preview = None
        if raw_output is not None:
            preview = json.dumps(raw_output, ensure_ascii=False, default=str)
            if len(preview) > TOOL_OUTPUT_PREVIEW_LENGTH:
                preview = preview[:TOOL_OUTPUT_PREVIEW_LENGTH] + "..."

        chat_repository.insert_tool_call(
            session_id=session_id,
            user_id=user_id,
            assistant_message_id=saved["message_id"],
            tool_name=record["tool_name"],
            tool_input=record["tool_input"],
            tool_output_preview=preview,
            success=record["success"],
            error_message=record["error_message"],
        )

    return {
        "session_id": session_id,
        "message_id": saved["message_id"],
        "answer": answer,
        "tools": used_tools,
        "sources": collected_sources,
        "created_at": saved["created_at"],
    }


def run_tool_chat_manual(
    user_id: int,
    session_id: int,
    message: str,
    explanation_level: str = DEFAULT_EXPLANATION_LEVEL,
) -> dict:
    """bind_tools + 수동 while 루프로 Tool Calling을 처리합니다.

    1. LLM에 Tool 목록을 알려주고 1차 호출한다.
    2. tool_calls가 없으면 그대로 최종 답변이다 (일반 대화).
    3. tool_calls가 있으면 Tool을 직접 실행하고, 결과를 ToolMessage로 넣어 재호출한다.
    4. tool_calls가 없어지거나 MAX_TOOL_ITERATIONS에 도달할 때까지 반복한다.
    """
    history = _load_recent_history(session_id)
    _maybe_generate_title(session_id, history, message)
    chat_repository.insert_message(session_id, user_id, "user", message)

    tools = build_tools_for_user(user_id, session_id)
    tools_by_name = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=_build_system_prompt(explanation_level)),
        *history,
        HumanMessage(content=message),
    ]

    used_tools: list[dict] = []
    collected_sources: list[dict] = []

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    iterations = 0
    while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        iterations += 1

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = tools_by_name.get(tool_name)

            error_message = None
            if tool is None:
                tool_result = {"error": f"알 수 없는 Tool입니다: {tool_name}"}
                error_message = tool_result["error"]
            else:
                try:
                    tool_result = tool.invoke(tool_args)
                    if isinstance(tool_result, dict):
                        error_message = tool_result.get("error")
                except Exception as exc:
                    tool_result = {"error": str(exc)}
                    error_message = str(exc)

            used_tools.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_args,
                    "success": error_message is None,
                    "error_message": error_message,
                    "raw_output": tool_result,
                }
            )

            if isinstance(tool_result, dict) and tool_result.get("sources"):
                collected_sources.extend(tool_result["sources"])

            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False, default=str),
                    tool_call_id=tool_call["id"],
                )
            )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    answer = response.content

    return _finalize_turn(session_id, user_id, answer, used_tools, collected_sources)


def run_tool_chat_agent(
    user_id: int,
    session_id: int,
    message: str,
    explanation_level: str = DEFAULT_EXPLANATION_LEVEL,
) -> dict:
    """create_tool_calling_agent + AgentExecutor로 Tool Calling을 처리합니다.

    manual 방식과 Tool 판단 능력 자체는 동일하지만, tool_calls를 직접 반복 처리하는 대신
    LangChain이 제공하는 Agent 실행기가 그 반복(ReAct 스타일 루프)을 대신 해줍니다.
    """
    history = _load_recent_history(session_id)
    _maybe_generate_title(session_id, history, message)
    chat_repository.insert_message(session_id, user_id, "user", message)

    tools = build_tools_for_user(user_id, session_id)

    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _build_system_prompt(explanation_level)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=MAX_TOOL_ITERATIONS,
        return_intermediate_steps=True,
    )

    result = executor.invoke({"input": message, "chat_history": history})
    answer = result["output"]

    used_tools: list[dict] = []
    collected_sources: list[dict] = []
    for action, observation in result.get("intermediate_steps", []):
        error_message = None
        if isinstance(observation, dict):
            error_message = observation.get("error")
            if observation.get("sources"):
                collected_sources.extend(observation["sources"])

        used_tools.append(
            {
                "tool_name": action.tool,
                "tool_input": action.tool_input,
                "success": error_message is None,
                "error_message": error_message,
                "raw_output": observation,
            }
        )

    return _finalize_turn(session_id, user_id, answer, used_tools, collected_sources)

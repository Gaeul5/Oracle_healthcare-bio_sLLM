"""약물 정보 QA 에이전트 그래프 (ReAct 패턴 + openFDA 조회 도구).

다이어그램
----------
    START -> agent --tool_calls 있음--> tools -> agent (반복)
                  \\--tool_calls 없음--> END
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = (
    "너는 약물/건강기능식품 정보 QA 어시스턴트다. 다음 도구를 상황에 맞게 사용해라:\n"
    "- search_drug_info: 의약품의 효능/용법/주의사항 조회 (openFDA, 공식 데이터)\n"
    "- check_drug_interaction: 의약품 간 상호작용 조회 (openFDA, 공식 데이터)\n"
    "- search_supplement_info: 건강기능식품/영양제(예: 이노시톨) 일반 개요 조회 "
    "(Wikipedia, 참고용 비공식 정보)\n"
    "약물명은 도구에 넘기기 전에 영문 성분명으로 변환해라.\n"
    "\n"
    "매우 중요한 원칙 - 상호작용/안전성에 대해 절대 추측하지 마라:\n"
    "- check_drug_interaction 결과에 상대 약물이 명시적으로 언급되지 않으면, "
    "'공식 데이터베이스에 해당 조합의 상호작용 정보가 없습니다'라고 명확히 답해라. "
    "안심시키는 답을 지어내지 마라.\n"
    "- 의약품과 건강기능식품(예: 이노시톨) 조합은 check_drug_interaction으로 검증할 수 없다. "
    "search_supplement_info로 성분 자체에 대한 일반 정보는 제공하되, "
    "그 조합이 안전한지는 절대 판단하지 말고 반드시 의사/약사 상담을 안내해라.\n"
    "\n"
    "답변 마지막에는 항상 '실제 복용 전 의사/약사와 상담하세요.'를 덧붙여라."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(
    model="gemma-4-e4b-it",
    base_url="http://211.51.63.154:58500/v1",
    api_key="not-needed",
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "end"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

app = builder.compile()


if __name__ == "__main__":
    print("=" * 80)
    print("약물 정보 QA 챗봇 (openFDA 기반). 종료: quit/exit")
    print("=" * 80)

    history: list[BaseMessage] = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "종료"):
            break
        if not question:
            continue

        history.append(HumanMessage(content=question))
        result = app.invoke({"messages": history})
        history = result["messages"]
        print(f"\nAI: {history[-1].content}")

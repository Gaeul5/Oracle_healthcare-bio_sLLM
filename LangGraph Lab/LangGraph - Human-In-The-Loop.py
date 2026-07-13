import os
import operator
import json
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 환경 설정 ---
load_dotenv()
print("=" * 80)
print("LangGraph를 사용한 Tool-Calling Agent 시스템 (Human-in-the-loop)")
print("=" * 80)


# =================================================================
#  Part 1: 도구(Tools) 정의
# =================================================================
print("\n[Part 1] 도구(Tools) 정의")
print("-" * 80)

@tool
def google_search(query: str) -> str:
    """인터넷에서 정보를 검색합니다. 최신 정보나 실시간 데이터가 필요할 때 사용하세요."""
    print(f"--- 도구 실행: google_search (쿼리: {query}) ---")
    search = GoogleSerperAPIWrapper()
    try:
        result = search.run(query)
        return f"검색 결과: {result}"
    except Exception as e:
        return f"검색 중 오류 발생: {e}"

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 2' 또는 '10 * 5'"""
    print(f"--- 도구 실행: calculator (표현식: {expression}) ---")
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"

tools = [google_search, calculator]
tool_node = ToolNode(tools)

print("도구 'google_search', 'calculator'가 정의되었습니다.")


# ============================================================
#  Part 2: LangGraph 설계 (상태, 노드, 엣지 정의)
# ============================================================
print("\n[Part 2] LangGraph 설계")
print("-" * 80)

# --- 2-1. 그래프 상태 (State) 정의 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 2-2. 에이전트 및 노드 정의 ---
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """에이전트 노드: 다음 행동(도구 사용 또는 답변)을 결정합니다."""
    print("--- 노드 실행: agent_node ---")
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

def human_review_node(state: AgentState):
    """
    사람 검토 노드: 에이전트의 도구 사용 결정을 사람이 검토하고 수정합니다.
    """
    print("--- 노드 실행: human_review_node ---")
    last_message = state['messages'][-1]
    
    # tool_calls가 없으면 아무것도 하지 않고 다음으로 넘어감
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    original_tool_calls = last_message.tool_calls
    modified_tool_calls = []
    
    for tool_call in original_tool_calls:
        # 'google_search' 도구에 대해서만 개입
        if tool_call['name'] == 'google_search':
            query = tool_call['args']['query']
            print("\n--------------------------------------------------")
            print(f"에이전트가 다음 검색을 제안했습니다:")
            print(f"   검색어: '{query}'")
            print("--------------------------------------------------")
            
            feedback = input("승인하시겠습니까? (Enter: 승인, n: 취소, 다른 내용 입력: 검색어 수정): ").strip()

            if feedback.lower() == 'n':
                print("   -> 사용자가 검색을 취소했습니다.")
                continue # 이 도구 호출은 건너뜀
            elif feedback:
                print(f"   -> 사용자가 검색어를 수정했습니다: '{feedback}'")
                tool_call['args']['query'] = feedback
        
        modified_tool_calls.append(tool_call)

    # 마지막 AIMessage를 수정된 tool_calls로 업데이트
    # 상태를 직접 수정하는 대신, 수정된 메시지를 포함한 새로운 상태를 반환해야 함
    # LangGraph는 상태 업데이트를 병합하므로, 기존 메시지를 다시 추가할 필요는 없음
    if not modified_tool_calls:
        # 모든 도구 호출이 취소된 경우
        last_message.tool_calls = []
        # 사용자에게 피드백을 주기 위한 메시지 추가
        return {"messages": [ToolMessage(content="사용자가 모든 작업을 취소했습니다.", tool_call_id="human_intervention")]}
    else:
        last_message.tool_calls = modified_tool_calls
        return {}


# --- 2-3. 조건부 엣지(Edge) 함수 정의 ---
def should_continue(state: AgentState) -> str:
    """에이전트의 응답에 따라 다음 경로를 결정합니다."""
    print("--- 조건부 엣지 실행: should_continue ---")
    last_message = state['messages'][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 도구 호출이 있으면 사람의 검토를 받으러 감
        print("   (결정) 도구 호출 필요 -> human_review_node로 이동")
        return "review"
    else:
        print("   (결정) 답변 완료 -> END")
        return "end"

# --- 2-4. 그래프 생성 및 연결 ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("human_review", human_review_node) # 새로운 노드
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "review": "human_review", # 도구 호출 시 human_review로
        "end": END,
    },
)
workflow.add_edge("human_review", "action") # 검토가 끝나면 도구 실행
workflow.add_edge("action", "agent")

app = workflow.compile()
print("\nLangGraph 컴파일 완료!")
try:
    img_bytes = app.get_graph().draw_mermaid_png()
    with open("agent_graph_with_human.png", "wb") as f:
        f.write(img_bytes)
    print("그래프 구조를 'agent_graph_with_human.png' 파일로 저장했습니다.")
except Exception as e:
    print(f"그래프 시각화 실패 (Graphviz 필요): {e}")


# ============================================================
#  Part 3: LangGraph Agent 챗봇 실행
# ============================================================
print("\n" + "=" * 80)
print("대화형 Agent 시스템을 시작합니다.")
print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
print("=" * 80)

while True:
    user_question = input("\nYou: ").strip()
    if user_question.lower() in ['quit', 'exit', '종료']:
        print("\n시스템을 종료합니다.")
        break
    if not user_question:
        continue

    inputs = {"messages": [HumanMessage(content=user_question)]}
    try:
        for event in app.stream(inputs, stream_mode="values"):
            final_message = event["messages"][-1]
            if not (hasattr(final_message, "tool_calls") and final_message.tool_calls):
                 print(f"\nAI: {final_message.content}")
        print("-" * 80)

    except Exception as e:
        print(f"\n[오류 발생] {e}")
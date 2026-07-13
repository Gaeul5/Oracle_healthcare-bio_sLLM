from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

# ===== 1단계: 상태 정의 =====
# 그래프가 기억할 정보의 구조를 정의합니다
class CalculatorState(TypedDict):
    total: int        # 현재까지의 총합
    history: list     # 계산 기록

# ===== 2단계: 노드 함수 정의 =====
# 실제 계산을 수행하는 함수입니다
def add_number(state: CalculatorState) -> dict:
    """새 숫자를 더하고 기록을 남깁니다"""
    current_total = state["total"]
    number_to_add = 10  # 예제에서는 항상 10을 더함

    new_total = current_total + number_to_add
    new_history = state["history"] + [f"{current_total} + {number_to_add} = {new_total}"]

    print(f"📊 계산: {current_total} + {number_to_add} = {new_total}")

    return {
        "total": new_total,
        "history": new_history
    }

# ===== 3단계: 그래프 구성 =====
graph = StateGraph(CalculatorState)
graph.add_node("add", add_number)
graph.add_edge(START, "add")
graph.add_edge("add", END)

# ===== 4단계: 메모리 연결 =====
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)

# ===== 5단계: 실행 =====
config = {"configurable": {"thread_id": "calculator_session"}}

# 첫 번째 계산 - 초기 상태 제공
print("=== 첫 번째 계산 ===")
result = app.invoke({"total": 0, "history": []}, config=config)
print(f"결과: total={result['total']}, history={result['history']}\n")

# 두 번째 계산 - 이전 상태 자동 복원!
print("=== 두 번째 계산 ===")
result = app.invoke({}, config=config)   # 모든 값 생략 가능!
print(f"결과: total={result['total']}, history={result['history']}\n")

# 세 번째 계산
print("=== 세 번째 계산 ===")
result = app.invoke({}, config=config)  
print(f"결과: total={result['total']}, history={result['history']}")
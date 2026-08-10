from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict


# ===== 1단계: 상태 정의 =====
class CalculatorState(TypedDict):
    total: int
    history: list


# ===== 로그 출력 함수 =====
def log_node_start(node_name: str, state: CalculatorState):
    print(f"\n[{node_name}] 실행 시작")
    print(f"   - 현재 total : {state['total']}")
    print(f"   - 현재 history 개수 : {len(state['history'])}")


def log_node_end(node_name: str, result: dict):
    print(f"[{node_name}] 실행 완료")
    print(f"   - 변경 total : {result['total']}")
    print(f"   - 추가 기록 : {result['history'][-1]}")


def print_current_state(app, config):
    state = app.get_state(config)

    print("\n==============================")
    print("현재 최신 체크포인트 상태")
    print("==============================")
    print(f"현재 값(values): {state.values}")
    print(f"다음 실행 노드(next): {state.next if state.next else 'END'}")
    print(f"checkpoint_id: {state.config['configurable'].get('checkpoint_id')}")


def print_checkpoint_history(app, config):
    history = list(app.get_state_history(config))

    print("\n==============================")
    print("체크포인트 히스토리")
    print("==============================")

    # get_state_history는 보통 최신 상태부터 나오므로,
    # 강의에서는 오래된 순서로 보이게 reversed 처리
    for idx, snapshot in enumerate(reversed(history), start=1):
        next_node = snapshot.next if snapshot.next else "END"
        checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")

        print(f"\n[{idx}] 체크포인트")
        print(f"   - 다음 실행 노드 : {next_node}")
        print(f"   - 저장된 상태 : {snapshot.values}")
        print(f"   - checkpoint_id : {checkpoint_id}")


# ===== 2단계: 노드 함수 정의 =====
def add_10(state: CalculatorState) -> dict:
    node_name = "add_10"
    log_node_start(node_name, state)

    current_total = state["total"]
    new_total = current_total + 10

    result = {
        "total": new_total,
        "history": state["history"] + [f"{current_total} + 10 = {new_total}"]
    }

    log_node_end(node_name, result)
    return result


def multiply_2(state: CalculatorState) -> dict:
    node_name = "multiply_2"
    log_node_start(node_name, state)

    current_total = state["total"]
    new_total = current_total * 2

    result = {
        "total": new_total,
        "history": state["history"] + [f"{current_total} * 2 = {new_total}"]
    }

    log_node_end(node_name, result)
    return result


def subtract_5(state: CalculatorState) -> dict:
    node_name = "subtract_5"
    log_node_start(node_name, state)

    current_total = state["total"]
    new_total = current_total - 5

    result = {
        "total": new_total,
        "history": state["history"] + [f"{current_total} - 5 = {new_total}"]
    }

    log_node_end(node_name, result)
    return result


# ===== 3단계: 그래프 구성 =====
graph = StateGraph(CalculatorState)

graph.add_node("add_10", add_10)
graph.add_node("multiply_2", multiply_2)
graph.add_node("subtract_5", subtract_5)

graph.add_edge(START, "add_10")
graph.add_edge("add_10", "multiply_2")
graph.add_edge("multiply_2", "subtract_5")
graph.add_edge("subtract_5", END)


# ===== 4단계: 체크포인터 연결 =====
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)


# ===== 5단계: 실행 =====
config = {
    "configurable": {
        "thread_id": "calculator_session"
    }
}

print("\n그래프 실행 시작")
result = app.invoke(
    {
        "total": 0,
        "history": []
    },
    config=config
)

print("\n최종 결과")
print(result)


# ===== 6단계: 체크포인트 로그 확인 =====
print_current_state(app, config)
print_checkpoint_history(app, config)
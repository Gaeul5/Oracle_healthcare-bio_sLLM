from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

# 상태 정의
class VisitorState(TypedDict):
    name: str           # 방문자 이름
    visit_count: int    # 방문 횟수
    message: str        # 환영 메시지

# 노드 함수: 방문자 맞이하기
def welcome_visitor(state: VisitorState) -> dict:
    """방문 횟수에 따라 다른 인사를 합니다"""
    name = state["name"]
    count = state["visit_count"] + 1

    # 방문 횟수에 따른 메시지 분기
    if count == 1:
        message = f"🎉 환영합니다, {name}님! 처음 오셨네요!"
    elif count < 5:
        message = f"👋 반갑습니다, {name}님! ({count}번째 방문)"
    else:
        message = f"⭐ {name}님, 단골이시네요! ({count}번째 방문)"

    return {"visit_count": count, "message": message}

# 그래프 구성
graph = StateGraph(VisitorState)
graph.add_node("welcome", welcome_visitor)
graph.add_edge(START, "welcome")
graph.add_edge("welcome", END)

# 메모리 설정
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)

# ===== 서로 다른 사용자 시뮬레이션 =====

# Alice 설정 (thread_id로 구분)
alice_config = {"configurable": {"thread_id": "user_alice"}}

# Bob 설정
bob_config = {"configurable": {"thread_id": "user_bob"}}

print("=" * 50)
print("Alice 첫 방문")
result = app.invoke({"name": "Alice", "visit_count": 0, "message": ""}, config=alice_config)
print(result["message"])

print("\n" + "=" * 50)
print("Bob 첫 방문")
result = app.invoke({"name": "Bob", "visit_count": 0, "message": ""}, config=bob_config)
print(result["message"])

print("\n" + "=" * 50)
print("Alice 두 번째 방문")
# name만 전달 - visit_count는 메모리에서 자동 복원!
result = app.invoke({"name": "Alice", "message": ""}, config=alice_config)
print(result["message"])

print("\n" + "=" * 50)
print("Alice 세 번째 ~ 다섯 번째 방문")
for i in range(3):
    result = app.invoke({"name": "Alice", "message": ""}, config=alice_config)
    print(result["message"])
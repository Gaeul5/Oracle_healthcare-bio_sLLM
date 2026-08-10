from typing import TypedDict
from langgraph.graph import StateGraph

# --- 1. 상태(State) 정의 ---
class GraphState(TypedDict):
    """
    그래프의 상태를 나타냅니다.
    Args:
        value (int): 0에서 시작하여 1씩 증가할 카운터 값
    """
    value: int

# --- 2. 노드(Node) 함수 정의 ---
def add_one(state: GraphState):
    """
    현재 상태의 'value' 값에 1을 더합니다.
    """
    current_value = state['value']
    print(f"--- 'add_one' 노드 실행 ---")
    print(f"   (입력) 현재 상태 'value': {current_value}")
    
    new_value = current_value + 1
    
    print(f"   (출력) 'value'를 {new_value}로 업데이트")
    
    return {"value": new_value}

# --- 3. 조건부 엣지(Edge) 함수 정의 ---
def should_continue(state: GraphState):
    """
    상태의 'value' 값을 확인하여 다음 단계를 결정합니다.
    """
    current_value = state['value']
    print(f"\n--- 'should_continue' 조건부 엣지 실행 ---")
    print(f"   (상태 확인) 현재 'value': {current_value}")
    
    if current_value < 3:
        print(f"   (결정) 값이 3보다 작으므로 'add_one' 노드로 다시 이동")
        return "continue_adding" 
    else:
        print(f"   (결정) 값이 3 이상이므로 그래프 종료")
        return "end_graph"

# --- 4. 그래프 생성 및 노드/엣지 연결 ---
workflow = StateGraph(GraphState)
workflow.add_node("adder_node", add_one)
workflow.set_entry_point("adder_node")

workflow.add_conditional_edges(
    "adder_node",
    should_continue,
    {
        "continue_adding": "adder_node",
        "end_graph": "__end__" 
    }
)

# --- 5. 그래프 컴파일 ---
app = workflow.compile()

# --- 6. 그래프 시각화 (VSCode용) ---
print("=" * 60)
print("그래프 구조 시각화")
print("=" * 60)

try:
    # 방법 1: Mermaid 텍스트 출력 (항상 작동)
    print("\n[Mermaid 다이어그램 텍스트]")
    mermaid_text = app.get_graph().draw_mermaid()
    print(mermaid_text)
    
    # Mermaid 텍스트를 파일로 저장
    with open("langgraph_diagram.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_text)
    print("\n✓ Mermaid 다이어그램이 'langgraph_diagram.mmd'로 저장되었습니다.")
    print("  (VSCode에서 Mermaid 확장 프로그램으로 열어보세요)")
    
    # 방법 2: PNG 이미지 생성 시도
    print("\n[PNG 이미지 생성 시도...]")
    graph_image = app.get_graph().draw_mermaid_png()
    
    # 이미지 파일로 저장
    with open("langgraph_visualization.png", "wb") as f:
        f.write(graph_image)
    print("✓ 그래프 이미지가 'langgraph_visualization.png'로 저장되었습니다.")
    
    # PIL로 이미지 열기 시도
    try:
        from PIL import Image
        img = Image.open("langgraph_visualization.png")
        img.show()
        print("✓ 기본 이미지 뷰어로 이미지를 열었습니다.")
    except ImportError:
        print("  (이미지를 보려면 'langgraph_visualization.png' 파일을 직접 여세요)")
    
except Exception as e:
    print(f"\n⚠ 이미지 생성 중 오류 발생: {e}")
    print("\n대신 텍스트 형식으로 그래프 구조를 표시합니다:")
    print(app.get_graph())

print("\n" + "=" * 60)

# --- 7. 그래프 실행 ---
print("\n--- 그래프 실행 시작 (초기 상태: {'value': 0}) ---")
final_state = app.invoke({"value": 0})

print("\n--- 그래프 실행 종료 ---")
print(f"최종 상태: {final_state}")
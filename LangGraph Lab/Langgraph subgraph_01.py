"""부모 그래프 안에서 서브그래프를 하나의 노드처럼 실행하는 예제.

실행 방법
---------
13. LangGraph 폴더에서 다음 명령을 실행한다.

    python .\07_subgraph_basic.py

핵심 개념
---------
1. 서브그래프도 StateGraph로 만들고 먼저 compile()한다.
2. 컴파일된 서브그래프를 부모 그래프의 add_node()에 전달한다.
3. 부모와 자식 상태에서 이름이 같은 키(request, result)는 서로 전달된다.
4. analysis, child_trace는 서브그래프 내부에서만 사용하는 비공개 상태다.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def log(scope: str, message: str) -> None:
    """부모/자식 실행 영역과 실행 스레드를 함께 표시한다."""
    print(
        f"[{time.strftime('%H:%M:%S')}] "
        f"[{threading.current_thread().name}] "
        f"[{scope}] {message}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 1. 서브그래프 상태와 노드 정의
# ---------------------------------------------------------------------------
class ChildState(TypedDict):
    # 부모 상태와 이름이 같으므로 부모 -> 서브그래프로 전달되는 공유 키다.
    request: str
    # 서브그래프가 작성하고 부모가 이어서 사용하는 공유 키다.
    result: str

    # 아래 두 키는 ParentState에 없으므로 서브그래프 내부에서만 사용한다.
    analysis: str
    child_trace: list[str]


def analyze_request(state: ChildState) -> dict:
    """첫 번째 자식 노드: 요청을 분석해 자식 전용 상태를 만든다."""
    log("CHILD", "analyze_request 노드 진입")
    log("CHILD", f"부모로부터 받은 공유 상태: request={state['request']!r}")
    log("CHILD", "현재 analysis와 child_trace는 자식 그래프 전용 키")

    update = {
        "analysis": f"요청 길이={len(state['request'])}, 핵심어=LangGraph",
        "child_trace": ["analyze_request 완료"],
    }
    log("CHILD", f"반환하는 부분 상태: {update}")
    return update


def generate_answer(state: ChildState) -> dict:
    """두 번째 자식 노드: 내부 분석 결과로 부모에게 돌려줄 결과를 만든다."""
    log("CHILD", "generate_answer 노드 진입")
    log("CHILD", f"앞 자식 노드에서 저장한 analysis={state['analysis']!r}")
    log("CHILD", f"현재 child_trace={state['child_trace']}")

    update = {
        # result는 부모와 공유되는 키이므로 서브그래프 종료 후 부모가 받는다.
        "result": f"처리 완료: {state['request']} ({state['analysis']})",
        "child_trace": state["child_trace"] + ["generate_answer 완료"],
    }
    log("CHILD", f"반환하는 부분 상태: {update}")
    return update


# 서브그래프 자체도 START와 END가 있는 완전한 그래프다.
child_builder = StateGraph(ChildState)
child_builder.add_node("analyze_request", analyze_request)
child_builder.add_node("generate_answer", generate_answer)
child_builder.add_edge(START, "analyze_request")
child_builder.add_edge("analyze_request", "generate_answer")
child_builder.add_edge("generate_answer", END)
child_graph = child_builder.compile()


# ---------------------------------------------------------------------------
# 2. 부모 그래프 상태와 노드 정의
# ---------------------------------------------------------------------------
class ParentState(TypedDict):
    request: str
    result: str
    parent_trace: list[str]


def prepare(state: ParentState) -> dict:
    """서브그래프에 들어가기 전 부모 노드."""
    log("PARENT", f"prepare 입력 상태: {state}")
    update = {"parent_trace": state["parent_trace"] + ["prepare 완료"]}
    log("PARENT", f"prepare 반환 상태: {update}")
    log("FLOW", "이제 컴파일된 child_graph 노드로 이동")
    return update


def finish(state: ParentState) -> dict:
    """서브그래프 실행이 끝난 뒤 다시 실행되는 부모 노드."""
    log("FLOW", "child_graph가 종료되어 부모 그래프로 제어권 복귀")
    log("PARENT", f"서브그래프에서 받은 공유 키 result={state['result']!r}")
    log("PARENT", f"부모의 기존 parent_trace도 유지됨={state['parent_trace']}")
    log(
        "PARENT",
        "ChildState 전용 키 analysis, child_trace는 ParentState에 없으므로 "
        "부모 상태 밖으로 노출되지 않음",
    )

    update = {"parent_trace": state["parent_trace"] + ["finish 완료"]}
    log("PARENT", f"finish 반환 상태: {update}")
    return update


parent_builder = StateGraph(ParentState)
parent_builder.add_node("prepare", prepare)

# 중요: compile된 서브그래프를 일반 함수 노드처럼 부모 그래프에 등록한다.
# 부모와 자식은 request, result라는 동일한 상태 키를 통해 데이터를 주고받는다.
parent_builder.add_node("child_graph", child_graph)

parent_builder.add_node("finish", finish)
parent_builder.add_edge(START, "prepare")
parent_builder.add_edge("prepare", "child_graph")
parent_builder.add_edge("child_graph", "finish")
parent_builder.add_edge("finish", END)
parent_graph = parent_builder.compile()


def show_graph_image() -> None:
    """부모와 서브그래프를 펼친 Mermaid와 PNG 이미지를 생성한다.

    ``xray=True``가 중요하다. 이 옵션이 없으면 child_graph가 하나의 상자로
    보이고, 옵션을 켜면 서브그래프 내부 노드까지 펼쳐서 보여준다.

    PNG 생성은 기본적으로 Mermaid 이미지 생성 서비스를 사용할 수 있으므로
    네트워크가 차단된 환경에서는 실패할 수 있다. Mermaid 원본(.mmd)은
    네트워크 없이도 항상 저장되며 VS Code Mermaid 확장 등으로 볼 수 있다.
    """
    output_dir = Path(__file__).resolve().parent
    mermaid_path = output_dir / "subgraph_diagram.mmd"
    png_path = output_dir / "subgraph_diagram.png"

    # xray=True: 부모 그래프 안의 서브그래프 내부 구조까지 재귀적으로 펼친다.
    drawable_graph = parent_graph.get_graph(xray=True)
    mermaid_text = drawable_graph.draw_mermaid()

    print("\n=== Mermaid 다이어그램 ===")
    print(mermaid_text)
    mermaid_path.write_text(mermaid_text, encoding="utf-8")
    print(f"\n[저장 완료] Mermaid 원본: {mermaid_path}")

    try:
        # Mermaid 문법을 PNG 바이트로 렌더링해 파일로 저장한다.
        png_bytes = drawable_graph.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"[저장 완료] 그래프 이미지: {png_path}")

        # Pillow가 설치되어 있으면 운영체제의 기본 이미지 뷰어로 바로 연다.
        try:
            from PIL import Image

            Image.open(png_path).show()
            print("[이미지 열기] 기본 이미지 뷰어로 그래프를 열었습니다.")
        except ImportError:
            print("[이미지 열기 생략] Pillow가 없어 PNG 파일만 저장했습니다.")
        except Exception as exc:
            print(f"[이미지 열기 실패] PNG 파일을 직접 열어주세요: {exc}")
    except Exception as exc:
        print(f"[PNG 생성 실패] {type(exc).__name__}: {exc}")
        print("Mermaid 원본은 정상 저장되었습니다. .mmd 파일을 열어 확인하세요.")


if __name__ == "__main__":
    print("\n=== LangGraph 서브그래프 기본 예제 ===")
    print("실행 구조:")
    print("  부모: START -> prepare -> [child_graph] -> finish -> END")
    print("  자식:                   START -> analyze -> generate -> END")
    print("공유 키: request, result")
    print("자식 전용 키: analysis, child_trace\n")

    # 실행 전에 전체 계층 구조를 Mermaid와 PNG로 시각화한다.
    show_graph_image()

    initial_state: ParentState = {
        "request": "LangGraph 서브그래프를 설명해 주세요",
        "result": "",
        "parent_trace": [],
    }
    print(f"초기 부모 상태: {initial_state}\n")

    final_state = parent_graph.invoke(initial_state)

    print("\n=== 실행 완료 ===")
    print(f"최종 부모 상태: {final_state}")
    print("\n확인 사항:")
    print("  1. result에는 서브그래프가 만든 값이 들어왔습니다.")
    print("  2. parent_trace는 서브그래프 실행 전후로 유지되었습니다.")
    print("  3. analysis와 child_trace는 최종 부모 상태에 나타나지 않습니다.")
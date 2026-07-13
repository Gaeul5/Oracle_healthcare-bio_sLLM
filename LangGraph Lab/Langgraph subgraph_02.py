"""두 개의 서브그래프를 병렬로 실행하고 부모에서 결과를 병합하는 예제.

실행
----
    python .\08_parallel_subgraphs.py

전체 구조
---------
                              ┌─ news_subgraph: search -> summarize ─┐
    START -> prepare -> 병렬 ─┤                                      ├─ merge -> END
                              └─ paper_subgraph: search -> summarize ┘

핵심 학습 내용
-------------
1. 각각의 서브그래프는 내부에 독립적인 상태와 여러 노드를 가진다.
2. 부모의 prepare에서 두 서브그래프로 edge를 연결하면 병렬 실행된다.
3. 두 서브그래프가 같은 부모 상태 키 results에 값을 반환한다.
4. 부모 State의 Reducer가 병렬 결과를 안전하게 하나의 리스트로 합친다.
5. 두 서브그래프가 모두 종료되어야 부모의 merge 노드가 실행된다.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


def log(scope: str, message: str) -> None:
    """실행 영역, 시간, 스레드 이름을 함께 출력한다."""
    print(
        f"[{time.strftime('%H:%M:%S')}] "
        f"[{threading.current_thread().name}] "
        f"[{scope}] {message}",
        flush=True,
    )


def merge_results(current: list[str], new: list[str]) -> list[str]:
    """부모 상태에서 두 서브그래프의 결과를 합치는 Reducer."""
    merged = current + new
    print("\n[PARENT REDUCER 호출] results 채널 병합", flush=True)
    print(f"  current (기존 누적값)       : {current}", flush=True)
    print(f"  new     (서브그래프 반환값) : {new}", flush=True)
    print(f"  merged  (병합 완료값)       : {merged}\n", flush=True)
    return merged


# ===========================================================================
# 1. 뉴스 조사 서브그래프
# ===========================================================================
class NewsState(TypedDict):
    # 부모와 공유하는 입력 키
    query: str
    # 부모와 공유하는 출력 키
    results: list[str]
    # 이 키는 뉴스 서브그래프 내부에서만 사용한다.
    news_raw: str


class SubgraphOutput(TypedDict):
    """두 서브그래프가 부모에게 공개하는 출력 인터페이스."""

    results: list[str]


def search_news(state: NewsState) -> dict:
    log("NEWS CHILD", f"search_news 시작, 공유 입력 query={state['query']!r}")
    log("NEWS CHILD", "뉴스 검색 작업 수행 중... (1.2초)")
    time.sleep(1.2)
    update = {"news_raw": f"{state['query']} 관련 최신 뉴스 원문"}
    log("NEWS CHILD", f"search_news 반환(자식 전용 상태): {update}")
    return update


def summarize_news(state: NewsState) -> dict:
    log("NEWS CHILD", f"summarize_news 입력 news_raw={state['news_raw']!r}")
    time.sleep(0.4)
    update = {"results": [f"[뉴스 요약] {state['query']} 도입 사례 증가"]}
    log("NEWS CHILD", f"부모와 공유할 results 반환: {update}")
    return update


def validate_news(state: NewsState) -> dict:
    log("NEWS CHILD", f"validate_news 입력 results={state['results']!r}")
    summary = state["results"][0]
    is_valid = len(summary) > 5
    log("NEWS CHILD", f"검증 결과: {'통과' if is_valid else '실패'}")
    if not is_valid:
        return {"results": [f"[검증 실패로 재작성] {summary}"]}
    return {}


# output_schema를 지정하면 query와 news_raw는 부모에게 반환되지 않고,
# 공개 출력인 results만 서브그래프 밖으로 전달된다.
news_builder = StateGraph(NewsState, output_schema=SubgraphOutput)
news_builder.add_node("search_news", search_news)
news_builder.add_node("summarize_news", summarize_news)
news_builder.add_node("validate_news", validate_news)
news_builder.add_edge(START, "search_news")
news_builder.add_edge("search_news", "summarize_news")
news_builder.add_edge("summarize_news", "validate_news")
news_builder.add_edge("validate_news", END)
news_subgraph = news_builder.compile()


# ===========================================================================
# 2. 논문 조사 서브그래프
# ===========================================================================
class PaperState(TypedDict):
    query: str
    results: list[str]
    # 이 키는 논문 서브그래프 내부에서만 사용한다.
    paper_raw: str


def search_papers(state: PaperState) -> dict:
    log("PAPER CHILD", f"search_papers 시작, 공유 입력 query={state['query']!r}")
    log("PAPER CHILD", "논문 검색 작업 수행 중... (0.8초)")
    time.sleep(0.8)
    update = {"paper_raw": f"{state['query']} 관련 학술 논문 원문"}
    log("PAPER CHILD", f"search_papers 반환(자식 전용 상태): {update}")
    return update


def summarize_papers(state: PaperState) -> dict:
    log("PAPER CHILD", f"summarize_papers 입력 paper_raw={state['paper_raw']!r}")
    time.sleep(0.5)
    update = {"results": [f"[논문 요약] {state['query']} 병렬 처리 성능 분석"]}
    log("PAPER CHILD", f"부모와 공유할 results 반환: {update}")
    return update


def validate_paper(state: PaperState) -> dict:
    log("PAPER CHILD", f"validate_paper 입력 results={state['results']!r}")
    summary = state["results"][0]
    is_valid = len(summary) > 5
    log("PAPER CHILD", f"검증 결과: {'통과' if is_valid else '실패'}")
    if not is_valid:
        return {"results": [f"[검증 실패로 재작성] {summary}"]}
    return {}


# paper_raw도 자식 내부에 감추고 results만 부모로 반환한다.
paper_builder = StateGraph(PaperState, output_schema=SubgraphOutput)
paper_builder.add_node("search_papers", search_papers)
paper_builder.add_node("summarize_papers", summarize_papers)
paper_builder.add_node("validate_paper", validate_paper)
paper_builder.add_edge(START, "search_papers")
paper_builder.add_edge("search_papers", "summarize_papers")
paper_builder.add_edge("summarize_papers", "validate_paper")
paper_builder.add_edge("validate_paper", END)
paper_subgraph = paper_builder.compile()


# ===========================================================================
# 3. 두 서브그래프를 병렬 실행하는 부모 그래프
# ===========================================================================
class ParentState(TypedDict):
    query: str
    # 병렬 서브그래프가 같은 키에 쓰므로 반드시 Reducer를 지정한다.
    # Reducer가 없으면 INVALID_CONCURRENT_GRAPH_UPDATE 오류가 발생한다.
    results: Annotated[list[str], merge_results]
    final_report: str


def prepare(state: ParentState) -> dict:
    log("PARENT", f"prepare 입력 상태={state}")
    log("FLOW", "prepare가 끝나면 news_subgraph와 paper_subgraph로 동시 fan-out")
    # query는 이미 초기 상태에 있으므로 변경할 상태가 없다.
    return {}


def merge(state: ParentState) -> dict:
    """두 서브그래프가 모두 끝난 뒤 실행되는 fan-in 노드."""
    log("FLOW", "두 서브그래프가 모두 종료됨 -> merge 노드 실행")
    log("PARENT", f"Reducer가 합친 results={state['results']}")

    report_lines = ["통합 조사 보고서"]
    report_lines.extend(f"- {item}" for item in state["results"])
    update = {"final_report": "\n".join(report_lines)}
    log("PARENT", f"merge 반환 부분 상태={update}")
    return update


parent_builder = StateGraph(ParentState)
parent_builder.add_node("prepare", prepare)

# 컴파일된 두 서브그래프를 부모의 노드로 등록한다.
parent_builder.add_node("news_subgraph", news_subgraph)
parent_builder.add_node("paper_subgraph", paper_subgraph)
parent_builder.add_node("merge", merge)

parent_builder.add_edge(START, "prepare")

# 한 노드에서 여러 edge로 나가므로 두 서브그래프는 같은 superstep에서 병렬 실행된다.
parent_builder.add_edge("prepare", "news_subgraph")
parent_builder.add_edge("prepare", "paper_subgraph")

# LangGraph는 두 서브그래프가 모두 끝난 다음 merge를 한 번만 실행한다.
parent_builder.add_edge("news_subgraph", "merge")
parent_builder.add_edge("paper_subgraph", "merge")
parent_builder.add_edge("merge", END)
parent_graph = parent_builder.compile()


def show_graph_image() -> None:
    """서브그래프 내부까지 펼친 Mermaid와 PNG 이미지를 생성한다."""
    output_dir = Path(__file__).resolve().parent
    mermaid_path = output_dir / "parallel_subgraphs_diagram.mmd"
    png_path = output_dir / "parallel_subgraphs_diagram.png"

    # xray=True가 두 서브그래프 내부의 search/summarize 노드를 펼쳐 보여준다.
    drawable_graph = parent_graph.get_graph(xray=True)
    mermaid_text = drawable_graph.draw_mermaid()

    print("\n=== Mermaid 다이어그램 ===")
    print(mermaid_text)
    mermaid_path.write_text(mermaid_text, encoding="utf-8")
    print(f"[저장 완료] Mermaid 원본: {mermaid_path}")

    try:
        png_path.write_bytes(drawable_graph.draw_mermaid_png())
        print(f"[저장 완료] 그래프 이미지: {png_path}")
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
        print("Mermaid 원본(.mmd)은 저장되었으므로 해당 파일로 확인할 수 있습니다.")


if __name__ == "__main__":
    print("\n=== 병렬 서브그래프 예제 ===")
    print("부모 구조:")
    print("                         ┌─ news_subgraph ──┐")
    print("  START -> prepare ──────┤                   ├─> merge -> END")
    print("                         └─ paper_subgraph ─┘")
    print("자식 내부:")
    print("  news_subgraph : search_news -> summarize_news")
    print("  paper_subgraph: search_papers -> summarize_papers\n")

    show_graph_image()

    initial_state: ParentState = {
        "query": "LangGraph",
        "results": [],
        "final_report": "",
    }
    print(f"\n초기 부모 상태: {initial_state}\n")
    final_state = parent_graph.invoke(initial_state)

    print("\n=== 실행 완료 ===")
    print(f"최종 results: {final_state['results']}")
    print(f"\n{final_state['final_report']}")
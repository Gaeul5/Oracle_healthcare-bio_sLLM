"""헬스케어 AI 연구 -> 취업 정보 Q&A, 부족하면 웹 검색으로 보강하는 Corrective RAG 예제.

다이어그램 구조
--------------
    START -> load_seed_document -> generate_answer -> (추가 정보 필요?)
                    ^                                    |
                    |                              YES ──┤── NO -> END
                    |                                    v
              add_document <- web_search <- generate_search_query

핵심 학습 내용 (오늘 배운 것 중 여기서 써먹을 것들)
--------------------------------------------
1. 조건부 엣지(add_conditional_edges)로 "추가 정보 필요?" YES/NO 분기.
2. add_document -> generate_answer로 되돌아가는 사이클(loop) 구성.
3. Reducer(operator.add)로 documents 리스트를 누적.
4. 도구 호출(google_search)로 검색 실행 -> Human-in-the-loop 파일의 tool 패턴 재사용 가능.
5. (선택) human_review 노드를 web_search 앞에 끼워 넣어 검색 승인/취소 받기.

TODO 표시된 부분만 채우면 됩니다.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

MAX_SEARCH_COUNT = 2

load_dotenv()

# ===========================================================================
# 0. 시드 문서 (다이어그램의 "문서" 박스 텍스트)
# ===========================================================================
SEED_DOCUMENT = """\
헬스케어는 데이터가 구조적으로 복잡한 도메인이야. EHR(전자건강기록), 의료영상,
시계열 바이오시그널, 다중모달 데이터를 동시에 다뤄야 하는 경우가 많아서,
데이터 전처리·결측치 처리·도메인 지식 기반 피처 엔지니어링 능력이 자연스럽게 쌓여.
이건 다른 도메인 연구자보다 확실히 강점이 돼.

규제 환경(FDA, MFDS, HIPAA 등)에서 모델 해석 가능성, 공정성, 불확실성 정량화를
다루다 보면 MLOps나 모델 거버넌스 관련 역량도 생겨. 요즘 기업에서 점점
중요하게 보는 부분이야.
"""


# ===========================================================================
# 1. 상태 정의
# ===========================================================================
class RAGState(TypedDict):
    question: str
    # 사이클을 돌 때마다 검색 결과 문서가 누적되므로 Reducer가 필요하다.
    documents: Annotated[List[str], operator.add]
    answer: str
    search_query: str
    search_result: str
    # 무한 루프 방지용 카운터
    search_count: int


# ===========================================================================
# 2. LLM 준비
# ===========================================================================
# TODO: 모델명 확인 (지난번 파일에서 429/모델명 이슈가 있었으니 실제 사용 가능한
#       모델인지 먼저 확인하세요)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ===========================================================================
# 3. 노드 정의
# ===========================================================================
def load_seed_document(state: RAGState) -> dict:
    """다이어그램의 '문서' 박스: 최초 1회만 시드 문서를 상태에 넣는다."""
    print("--- 노드 실행: load_seed_document ---")
    return {"documents": [SEED_DOCUMENT]}


def generate_answer(state: RAGState) -> dict:
    """다이어그램의 '1번 LLM 답변' 박스."""
    print("--- 노드 실행: generate_answer ---")
    context = "\n\n".join(
        f"[문서 {i + 1}]\n{doc}" for i, doc in enumerate(state["documents"])
    )
    prompt = f"""아래 문서들을 참고하여 질문에 답하세요.
여러 문서 중 하나라도 질문과 관련된 정보가 있으면 반드시 그 정보를 활용해 답변하세요.
모든 문서를 확인해도 관련 정보가 전혀 없을 때만 "문서에 없습니다"라고 답하세요.

문서:
{context}

질문: {state['question']}

답변:"""
    response = llm.invoke(prompt)
    answer = response.content
    print(f"   (생성된 답변) {answer}")
    return {"answer": answer}


def should_search_more(state: RAGState) -> str:
    """조건부 엣지 함수: '추가 정보가 필요합니까?' 다이아몬드."""
    print("--- 조건부 엣지 실행: should_search_more ---")

    if state.get("search_count", 0) >= MAX_SEARCH_COUNT:
        print(f"   (판정) 최대 검색 횟수({MAX_SEARCH_COUNT}) 도달 -> 종료")
        return "end"

    judge_prompt = f"""아래 답변이 질문에 필요한 정보를 충분히 담고 있는지 판단하세요.
"문서에 없습니다" 등 정보가 부족하다는 표현이 있으면 "부족"이라고만 답하세요.
충분하면 "충분"이라고만 답하세요.

질문: {state['question']}
답변: {state['answer']}

판정:"""
    judgment = llm.invoke(judge_prompt).content.strip()
    print(f"   (판정) {judgment}")
    return "search" if "부족" in judgment else "end"


def generate_search_query(state: RAGState) -> dict:
    """다이어그램의 '2번 LLM: 검색 쿼리 작성' 박스."""
    print("--- 노드 실행: generate_search_query ---")
    prompt = f"""아래 질문에 대해 문서에서 찾지 못한 정보를 웹에서 검색하려고 합니다.
검색에 사용할 간결한 검색어 한 줄만 작성하세요 (설명 없이 검색어만).

질문: {state['question']}
현재까지의 답변(부족한 부분 포함): {state['answer']}

검색어:"""
    query = llm.invoke(prompt).content.strip()
    print(f"   (생성된 검색어) {query}")
    return {"search_query": query}


def web_search(state: RAGState) -> dict:
    """다이어그램의 '검색 실행' 박스."""
    print(f"--- 노드 실행: web_search (검색어: {state['search_query']!r}) ---")
    search = GoogleSerperAPIWrapper()
    try:
        result = search.run(state["search_query"])
    except Exception as exc:
        result = f"검색 중 오류 발생: {exc}"
    return {"search_result": result}


def add_document(state: RAGState) -> dict:
    """다이어그램의 '문서 추가' 박스: 검색 결과를 documents에 누적한다."""
    print("--- 노드 실행: add_document ---")
    new_doc = f"[웹 검색 결과: {state['search_query']}]\n{state['search_result']}"
    return {
        "documents": [new_doc],
        "search_count": state.get("search_count", 0) + 1,
    }


# ===========================================================================
# 4. 그래프 연결
# ===========================================================================
builder = StateGraph(RAGState)

builder.add_node("load_seed_document", load_seed_document)
builder.add_node("generate_answer", generate_answer)
builder.add_node("generate_search_query", generate_search_query)
builder.add_node("web_search", web_search)
builder.add_node("add_document", add_document)

builder.add_edge(START, "load_seed_document")
builder.add_edge("load_seed_document", "generate_answer")

builder.add_conditional_edges(
    "generate_answer",
    should_search_more,
    {
        "search": "generate_search_query",
        "end": END,
    },
)

builder.add_edge("generate_search_query", "web_search")
builder.add_edge("web_search", "add_document")
# add_document -> generate_answer 로 되돌아가며 사이클을 형성한다.
# (다이어그램은 "문서" 박스로 되돌아가지만, 시드 문서를 다시 덮어쓰면 안 되므로
#  generate_answer로 직접 돌아가도록 설계했다. 다르게 하고 싶으면 바꿔도 된다.)
builder.add_edge("add_document", "generate_answer")

app = builder.compile()


# ===========================================================================
# 5. 실행부 (터미널에서 직접 질문 입력)
# ===========================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Corrective RAG 챗봇을 시작합니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 80)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "종료"):
            print("\n시스템을 종료합니다.")
            break
        if not question:
            continue

        initial_state: RAGState = {
            "question": question,
            "documents": [],
            "answer": "",
            "search_query": "",
            "search_result": "",
            "search_count": 0,
        }
        try:
            result = app.invoke(initial_state)
            print(f"\nAI: {result['answer']}")
            print(f"(참고: 문서 {len(result['documents'])}개, 검색 {result['search_count']}회)")
        except Exception as exc:
            print(f"\n[오류 발생] {exc}")
        print("-" * 80)

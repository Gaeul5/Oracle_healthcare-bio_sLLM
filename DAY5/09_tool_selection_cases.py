from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def search_documents(query: str, top_k: int = 3) -> str:
    """질문과 의미적으로 유사한 문서 chunk를 PGVector에서 검색합니다. 등록된 문서 내용을 근거로 답해야 할 때 사용합니다."""
    return "검색 결과 예시"


@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다. 사용자가 현재 시간이나 날짜를 물어볼 때 사용합니다."""
    return "현재 시간 예시"


@tool
def get_collection_stats() -> str:
    """현재 PGVector collection의 embedding row 개수와 source 개수를 반환합니다. collection 상태나 문서 저장 여부를 확인할 때 사용합니다."""
    return "collection 상태 예시"


@tool
def list_collections() -> str:
    """PGVector에 저장된 collection 목록을 보여줍니다. 어떤 collection이 저장되어 있는지 확인할 때 사용합니다."""
    return "collection 목록 예시"


TOOLS = [search_documents, get_current_time, get_collection_stats, list_collections]

TEST_CASES = [
    "안녕",
    "현재 시간을 알려줘",
    "컬렉션 상태를 보여줘",
    "컬렉션 목록을 보여줘",
    "생성형 AI 기술 동향을 설명해줘",
    "문서에서 AI 플러스 정책 관련 내용을 찾아줘",
]

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

system_instruction = (
    "너는 문서 기반 챗봇이다. 일반 대화도 가능하다. "
    "문서 검색이 필요하면 search_documents를 사용하고, 시간 질문은 get_current_time, "
    "컬렉션 상태는 get_collection_stats, 컬렉션 목록은 list_collections를 사용해라. "
    "Tool이 필요 없으면 그냥 답변해라."
)

print("=" * 80)
print("09. Tool 선택 사례 비교")
print("=" * 80)

for idx, question in enumerate(TEST_CASES, 1):
    print(f"\n[{idx}] 질문: {question}")
    response = llm_with_tools.invoke([HumanMessage(content=f"{system_instruction}\n\n질문: {question}")])
    if response.tool_calls:
        print("- Tool 호출 예정:")
        for tool_call in response.tool_calls:
            print(f"  * {tool_call['name']} | args={tool_call['args']}")
    else:
        print("- Tool 호출 없음")
        print(f"- 모델 답변: {response.content}")
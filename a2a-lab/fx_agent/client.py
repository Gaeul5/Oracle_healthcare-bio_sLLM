import asyncio

import httpx

from a2a.client import A2ACardResolver, ClientCallContext, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest

AGENT_URL = "http://127.0.0.1:9999"

TEST_QUERIES = [
    "100 USD를 KRW로",
    "50 EUR를 USD로",
    "10 JPY를 KRW로",
    "안녕하세요",
]


def extract_answer(chunk) -> str | None:
    """StreamResponse에서 최종 응답 텍스트만 뽑아냅니다 (없으면 None)."""
    if chunk.HasField("task") and chunk.task.artifacts:
        parts = chunk.task.artifacts[-1].parts
        if parts:
            return parts[0].text
    if chunk.HasField("artifact_update") and chunk.artifact_update.artifact.parts:
        return chunk.artifact_update.artifact.parts[0].text
    return None


async def send_message(text_query: str, *, verbose: bool = True) -> str | None:
    # 1) Agent Card 조회 → 이 에이전트가 어디서, 어떻게 응답하는지 확인
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=AGENT_URL)
        agent_card = await resolver.get_agent_card()

    # 2) 카드 정보를 바탕으로 클라이언트 생성 (여기선 스트리밍 없이 단순 호출)
    client = await create_client(agent=agent_card, client_config=ClientConfig(streaming=False))

    # 3) 메시지 전송
    message = new_text_message(text_query, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)

    # LLM 호출 + MCP 툴 호출이 더해져 기본 타임아웃(5초)보다 오래 걸릴 수 있어 넉넉히 설정
    context = ClientCallContext(timeout=60.0)

    if verbose:
        print(f"\n요청: '{text_query}'")

    answer = None
    async for chunk in client.send_message(request, context=context):
        found = extract_answer(chunk)
        if found is not None:
            answer = found

    if verbose:
        print(f"응답: {answer}")

    await client.close()
    return answer


async def run_test_queries() -> None:
    print("=" * 50)
    print("  FX Agent 호출 테스트")
    print("=" * 50)

    for query in TEST_QUERIES:
        await send_message(query)

    print("\n" + "=" * 50)
    print("  테스트 완료")
    print("=" * 50)


async def chat() -> None:
    print("=" * 50)
    print(f"  FX Agent 대화형 테스트 ({AGENT_URL})")
    print("  환율 변환 / 매출 조회를 자유롭게 물어보세요.")
    print("  종료하려면 quit 또는 exit 입력")
    print("=" * 50)

    while True:
        user_input = input("\n질문> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        answer = await send_message(user_input, verbose=False)
        print(f"답변> {answer}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        asyncio.run(chat())
    else:
        asyncio.run(run_test_queries())

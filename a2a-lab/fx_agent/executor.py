import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState

load_dotenv()

# ── 환율 표 (KRW 기준, 실제 서비스라면 외부 API를 호출) ─────────
RATES_TO_KRW = {
    "KRW": 1.0,
    "USD": 1350.0,
    "EUR": 1470.0,
    "JPY": 9.2,
}

# 오전에 만든 MCP 서버 (매출 DB 조회 툴, streamable-http)
MCP_SERVER_URL = "http://localhost:8000/mcp"

SYSTEM_PROMPT = (
    "너는 환율 변환과 매출 데이터 조회를 도와주는 에이전트다. "
    "통화 변환 질문에는 convert_currency 툴을 사용해라. "
    f"지원 통화: {', '.join(RATES_TO_KRW)}. "
    "매출/판매 데이터 질문에는 query 툴(매출 DB, sales 테이블: product, amount, sale_date)을 사용해라. "
    "둘 다 필요 없으면 툴 없이 바로 답변해라."
)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """한 통화 금액을 다른 통화로 환산합니다. 예: amount=100, from_currency='USD', to_currency='KRW'."""
    from_currency, to_currency = from_currency.upper(), to_currency.upper()
    if from_currency not in RATES_TO_KRW or to_currency not in RATES_TO_KRW:
        supported = ", ".join(RATES_TO_KRW)
        return f"지원하지 않는 통화입니다. 지원 통화: {supported}"

    krw = amount * RATES_TO_KRW[from_currency]
    result = krw / RATES_TO_KRW[to_currency]
    return f"{amount:g} {from_currency} = {result:,.2f} {to_currency}"


class FXAgent:
    """LangGraph 기반 두뇌: 환율 변환 + MCP 매출 DB 조회 툴을 상황에 맞게 선택해 호출"""

    def __init__(self) -> None:
        self._agent = None  # MCP 툴 로딩이 비동기라서 최초 호출 시 지연 생성

    async def _get_agent(self):
        if self._agent is None:
            mcp_client = MultiServerMCPClient(
                {
                    "sales-db": {
                        "transport": "streamable_http",
                        "url": MCP_SERVER_URL,
                    }
                }
            )
            mcp_tools = await mcp_client.get_tools()
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
            self._agent = create_agent(
                model=llm,
                tools=[convert_currency, *mcp_tools],
                system_prompt=SYSTEM_PROMPT,
            )
        return self._agent

    async def ainvoke(self, user_request: str) -> str:
        agent = await self._get_agent()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_request}]}
        )
        return result["messages"][-1].content


class FXAgentExecutor(AgentExecutor):
    """A2A 프로토콜과 FXAgent를 연결하는 실행기"""

    def __init__(self) -> None:
        self.agent = FXAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 1. 태스크 확보 (없으면 새로 생성)
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await task_updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("처리 중..."),
        )

        # 2. LangGraph 에이전트가 알맞은 툴을 선택해 처리
        query = get_message_text(context.message)
        result = await self.agent.ainvoke(query) if query else "요청 내용이 비어 있습니다."
        print(f"[FX Agent] '{query}' → {result}")

        # 3. 결과를 아티팩트로 전달 후 완료 처리
        await task_updater.add_artifact(
            parts=[new_text_part(text=result, media_type="text/plain")]
        )
        await task_updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("처리 완료!"),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")

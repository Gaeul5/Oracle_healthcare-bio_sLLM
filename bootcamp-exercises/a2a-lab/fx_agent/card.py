from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

AGENT_URL = "http://127.0.0.1:9999"

# ── 스킬 정의 : 이 에이전트가 "무엇을" 할 수 있는지 ──────────────
fx_skill = AgentSkill(
    id="fx_convert",
    name="환율 변환",
    description="한 통화 금액을 다른 통화로 환산합니다. 예: '100 USD를 KRW로'",
    input_modes=["text/plain"],
    output_modes=["text/plain"],
    tags=["a2a", "fx", "exchange-rate"],
    examples=["100 USD를 KRW로", "50 EUR를 USD로"],
)

sales_skill = AgentSkill(
    id="sales_query",
    name="매출 조회",
    description="MCP 매출 DB(sales 테이블)를 조회해 매출 데이터를 알려줍니다. 예: '노트북 매출 알려줘'",
    input_modes=["text/plain"],
    output_modes=["text/plain"],
    tags=["a2a", "mcp", "sales"],
    examples=["노트북 매출 알려줘", "전체 매출 합계는?"],
)

# ── 에이전트 카드 : "어디서, 어떻게" 접근하는지 + 위 스킬 목록 ──
public_agent_card = AgentCard(
    name="FX Agent",
    description="통화 간 환율을 변환해주는 A2A 에이전트",
    version="0.1.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url=AGENT_URL,
            protocol_version="1.0",
        )
    ],
    skills=[fx_skill, sales_skill],
)

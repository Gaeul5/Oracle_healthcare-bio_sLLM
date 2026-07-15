import asyncio
import uuid
import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Agent A (Orchestrator)")

AGENT_B_URL = "http://localhost:8001"

# ── A2A 데이터 모델 ─────────────────────────────────────────────

class UserRequest(BaseModel):
    message: str   # 사용자 입력

class AgentResponse(BaseModel):
    from_agent: str
    result: str

# ── Agent Card 조회 헬퍼 ────────────────────────────────────────
async def discover_agent(url: str) -> dict:
    """상대 Agent의 능력(Agent Card)을 조회합니다."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/.well-known/agent.json")
        return resp.json()

# ── Task 전송 헬퍼 ──────────────────────────────────────────────
async def send_task(agent_url: str, message: str) -> dict:
    """Agent에게 Task를 전송하고 결과를 받습니다."""
    task_payload = {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "content": message,
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{agent_url}/tasks/send",
            json=task_payload,
            timeout=10.0,
        )
        return resp.json()

# ── Agent Card 엔드포인트 ───────────────────────────────────────
@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "Agent A",
        "description": "사용자 요청을 분석해 적절한 에이전트에게 위임하는 오케스트레이터",
        "version": "1.0.0",
        "url": "http://localhost:8000",
    }

# ── 사용자 요청 처리 ────────────────────────────────────────────
@app.post("/ask", response_model=AgentResponse)
async def ask(req: UserRequest):
    print(f"\n[Agent A] 사용자 요청 수신: '{req.message}'")

    # 1) Agent B 능력 확인
    card = await discover_agent(AGENT_B_URL)
    print(f"[Agent A] Agent B 발견: {card['name']} - {card['description']}")

    # 2) Agent B에게 Task 위임
    print(f"[Agent A] → Agent B에게 Task 위임 중...")
    task_result = await send_task(AGENT_B_URL, req.message)

    print(f"[Agent A] ← Agent B 결과 수신: {task_result['result']}")
    return AgentResponse(from_agent=card["name"], result=task_result["result"])


# ── 시작 시 Agent B 자동 탐색 ──────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        card = await discover_agent(AGENT_B_URL)
        skills = [s["name"] for s in card["capabilities"]["skills"]]
        print(f"Agent B 연결 확인: {card['name']}")
        print(f"   지원 스킬: {', '.join(skills)}\n")
    except Exception:
        print("Agent B에 연결할 수 없습니다. agent_b.py를 먼저 실행하세요.\n")


if __name__ == "__main__":
    print("Agent A 시작 → http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
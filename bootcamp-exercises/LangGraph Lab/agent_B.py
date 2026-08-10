import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="Agent B (Worker)")

# ── A2A 데이터 모델 ─────────────────────────────────────────────

class Message(BaseModel):
    role: str        # "user" | "agent"
    content: str

class Task(BaseModel):
    id: str
    message: Message

class TaskResult(BaseModel):
    id: str
    status: str      # "completed" | "failed"
    result: Any

# ── Agent Card : A2A 필수 엔드포인트 ───────────────────────────
# 다른 에이전트가 "이 에이전트가 뭘 할 수 있지?" 를 알기 위해 조회
@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "Agent B",
        "description": "수학 계산과 날씨 조회를 처리하는 워커 에이전트",
        "version": "1.0.0",
        "capabilities": {
            "skills": [
                {
                    "id": "math_calc",
                    "name": "수학 계산",
                    "description": "사칙연산을 처리합니다. 예: '3 + 5', '10 * 4'",
                },
                {
                    "id": "weather_query",
                    "name": "날씨 조회",
                    "description": "도시 날씨를 조회합니다. 예: '서울 날씨'",
                },
            ]
        },
        "url": "http://localhost:8001",
    }

# ── Task 수신 & 처리 : A2A 핵심 엔드포인트 ────────────────────
@app.post("/tasks/send", response_model=TaskResult)
async def receive_task(task: Task):
    print(f"\n[Agent B] Task 수신 → id={task.id}, message='{task.message.content}'")

    content = task.message.content.strip()
    answer = _process(content)

    print(f"[Agent B] Task 처리 완료 → {answer}")
    return TaskResult(id=task.id, status="completed", result=answer)


def _process(content: str) -> str:
    """간단한 규칙 기반 처리 (실제 서비스에선 LLM 호출)"""

    # 날씨 조회
    WEATHER = {
        "서울": "13°C 맑음", "부산": "16°C 흐림",
        "제주": "18°C 비",   "도쿄": "15°C 맑음",
    }
    for city, weather in WEATHER.items():
        if city in content:
            return f"{city} 현재 날씨: {weather}"

    # 사칙연산 (예: "3 + 5", "10 * 4")
    for op in ["+", "-", "*", "/"]:
        if op in content:
            try:
                parts = content.replace("계산", "").strip().split(op)
                a, b = float(parts[0].strip()), float(parts[1].strip())
                result = {
                    "+": a + b, "-": a - b,
                    "*": a * b, "/": a / b if b != 0 else "0으로 나눌 수 없음",
                }[op]
                return f"{a} {op} {b} = {result}"
            except Exception:
                pass

    return f"처리할 수 없는 요청입니다: '{content}'"


if __name__ == "__main__":
    print("Agent B 시작 → http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
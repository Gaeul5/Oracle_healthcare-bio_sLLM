import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore

from card import public_agent_card
from executor import FXAgentExecutor

if __name__ == "__main__":
    # 1) 실행 로직(executor) + 태스크 저장소(task_store) + 카드(agent_card)를 묶는 핸들러
    request_handler = DefaultRequestHandler(
        agent_executor=FXAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=public_agent_card,
    )

    # 2) 카드 조회용 라우트 + JSON-RPC 요청 처리용 라우트를 조립
    routes = []
    routes.extend(create_agent_card_routes(public_agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, "/"))

    app = Starlette(routes=routes)

    print("FX Agent 시작 → http://127.0.0.1:9999")
    uvicorn.run(app, host="127.0.0.1", port=9999)

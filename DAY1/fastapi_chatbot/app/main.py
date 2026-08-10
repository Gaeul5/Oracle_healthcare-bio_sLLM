from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat_service


app = FastAPI(title="LangChain FastAPI Chatbot", version="1.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = chat_service.invoke(request.message, request.history, request.style)
    return ChatResponse(answer=result.answer)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        chat_service.stream(request.message, request.history, request.style),
        media_type="text/plain; charset=utf-8",
    )

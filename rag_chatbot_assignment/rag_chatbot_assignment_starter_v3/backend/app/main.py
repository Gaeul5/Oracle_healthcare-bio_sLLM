from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.db.connection import check_database_connection
from backend.app.routers import auth, chat, documents


app = FastAPI(title="LangChain RAG Chatbot Starter")

app.include_router(auth.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/api/health")
def health_check():
    db_ok, db_message = check_database_connection()
    missing_env = settings.missing_required_env()
    return {
        "ok": db_ok and not missing_env,
        "db_ok": db_ok,
        "db_message": db_message,
        "missing_env": missing_env,
    }


@app.get("/")
def index():
    return FileResponse(settings.FRONTEND_DIR / "index.html")


app.mount(
    "/css",
    StaticFiles(directory=str(settings.FRONTEND_DIR / "css")),
    name="css",
)

app.mount(
    "/js",
    StaticFiles(directory=str(settings.FRONTEND_DIR / "js")),
    name="js",
)
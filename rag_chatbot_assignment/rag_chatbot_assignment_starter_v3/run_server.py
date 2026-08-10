"""RAG 챗봇 스타터 실행 파일

실행 방법:
    python run_server.py

브라우저 접속:
    http://127.0.0.1:8000
"""

from pathlib import Path
import sys

import uvicorn


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

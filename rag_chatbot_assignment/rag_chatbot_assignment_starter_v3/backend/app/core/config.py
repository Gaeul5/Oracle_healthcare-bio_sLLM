from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"

# 루트 .env와 backend/.env 둘 다 지원합니다.
# 같은 값이 있으면 먼저 로드된 루트 .env 값을 우선 사용합니다.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env", override=False)


class Settings:
    PROJECT_ROOT: Path = PROJECT_ROOT
    BACKEND_DIR: Path = BACKEND_DIR
    FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
    UPLOAD_DIR: Path = BACKEND_DIR / "uploads"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "")

    # 강의 환경의 키 이름이 다를 수 있어서 대표 이름을 몇 개 허용합니다.
    # 실제 수업에서는 하나의 이름으로 통일하는 것을 권장합니다.
    OPENAI_EMBEDDING_MODEL: str = (
        os.getenv("OPENAI_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
        or "text-embedding-3-small"
    )

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

    # JWT_SECRET_KEY는 반드시 .env에서 설정해야 합니다.
    # 여기 fallback 값은 로컬 개발 편의용일 뿐이며, 실서비스에서는 절대 쓰면 안 됩니다.
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-only-insecure-secret")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

    def missing_required_env(self) -> list[str]:
        required = {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "OPENAI_MODEL": self.OPENAI_MODEL,
            "POSTGRES_HOST": self.POSTGRES_HOST,
            "POSTGRES_PORT": self.POSTGRES_PORT,
            "POSTGRES_DB": self.POSTGRES_DB,
            "POSTGRES_USER": self.POSTGRES_USER,
            "POSTGRES_PASSWORD": self.POSTGRES_PASSWORD,
        }
        return [name for name, value in required.items() if not value]


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

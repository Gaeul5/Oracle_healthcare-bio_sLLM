from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from backend.app.core.config import settings
from backend.app.db import auth_repository


class AuthError(Exception):
    """이메일 중복, 로그인 실패, 토큰 검증 실패 등 인증 관련 오류."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int) -> str:
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire_at}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """토큰을 검증하고 그 안에 담긴 user_id를 반환합니다. 유효하지 않으면 AuthError를 발생시킵니다."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AuthError("유효하지 않은 토큰입니다.") from exc

    return int(payload["sub"])


def register_user(email: str, password: str, name: str) -> dict:
    if auth_repository.find_user_by_email(email):
        raise AuthError("이미 가입된 이메일입니다.")

    password_hash = hash_password(password)
    return auth_repository.create_user(email=email, password_hash=password_hash, name=name)


def login_user(email: str, password: str) -> dict:
    user = auth_repository.find_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise AuthError("이메일 또는 비밀번호가 올바르지 않습니다.")

    access_token = create_access_token(user["id"])
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user["id"],
            "email": user["email"],
            "name": user["name"],
        },
    }

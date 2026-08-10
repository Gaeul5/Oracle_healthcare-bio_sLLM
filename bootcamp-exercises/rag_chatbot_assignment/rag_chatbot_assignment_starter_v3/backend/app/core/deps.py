from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.services.auth_service import AuthError, decode_access_token

# auto_error=False로 두면 토큰이 없을 때 FastAPI가 기본 403을 던지는 대신
# 아래에서 직접 401 + 우리 메시지로 응답할 수 있습니다.
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> int:
    """Authorization: Bearer <token> 헤더를 검증하고 현재 로그인한 user_id를 반환합니다.

    이 함수를 Depends()로 주입하면, 어떤 endpoint든 "지금 요청한 사람이 누구인지"를
    클라이언트가 아니라 서버가 토큰 검증을 통해 직접 확인하게 됩니다.

    HTTPBearer를 쓰면 OpenAPI에 security scheme으로 등록되어,
    Swagger UI(/docs) 화면 위쪽에 "Authorize" 버튼이 생깁니다.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="인증 토큰이 필요합니다.")

    try:
        return decode_access_token(credentials.credentials)
    except AuthError:
        raise HTTPException(status_code=401, detail="인증 토큰이 유효하지 않습니다.")

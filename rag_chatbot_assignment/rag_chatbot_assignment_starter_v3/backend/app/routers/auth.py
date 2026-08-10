from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.core.deps import get_current_user_id
from backend.app.db import auth_repository
from backend.app.services import auth_service
from backend.app.services.auth_service import AuthError

router = APIRouter(tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/auth/register")
def register(request: RegisterRequest):
    try:
        user = auth_service.register_user(
            email=request.email,
            password=request.password,
            name=request.name,
        )
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "user_id": user["id"],
        "email": user["email"],
        "name": user["name"],
    }


@router.post("/auth/login")
def login(request: LoginRequest):
    try:
        return auth_service.login_user(email=request.email, password=request.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@router.get("/me")
def get_me(current_user_id: int = Depends(get_current_user_id)):
    user = auth_repository.find_user_by_id(current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    return {
        "user_id": user["id"],
        "email": user["email"],
        "name": user["name"],
    }

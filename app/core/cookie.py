
from app.core.config import settings
from fastapi import Response


REFRESH_COOKIE_NAME = (
    "__Secure-refresh_token" if settings.COOKIE_SECURE else "refresh_token"
)

def set_refresh_cookie(response: Response,refresh_token: str):
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="none" if settings.COOKIE_SECURE else "lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE,
        path="/api/v1/auth"
    )

def clear_refresh_cookie(response: Response):
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="none" if settings.COOKIE_SECURE else "lax",
        path="/api/v1/auth"
    )
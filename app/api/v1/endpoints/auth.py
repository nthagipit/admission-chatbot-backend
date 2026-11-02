
import logging
from app.core.exception import AppException
from app.core.config import settings
from app.core.oauth import oauth
from app.schemas.auth_schema import RegisterRequestSchema
from fastapi import APIRouter, Depends, Request
from app.db.base import get_db
from app.services import auth_service
from sqlalchemy.orm import Session
from app.schemas.base_schema import DataResponse
from app.schemas.auth_schema import AuthResponseSchema, LoginRequestSchema

logger = logging.getLogger("app.api.v1.endpoints.auth")

router = APIRouter()

@router.post("/login", description="Login and get access token", response_model=DataResponse[AuthResponseSchema])
async def login(data: LoginRequestSchema, db: Session = Depends(get_db)):
    return auth_service.login(data, db)

@router.post("/register", description="Register new user", response_model=DataResponse[AuthResponseSchema])
async def register(data: RegisterRequestSchema, db: Session = Depends(get_db)):
    return auth_service.register(data, db)

@router.get("/google/login")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(
        request, 
        settings.GOOGLE_REDIRECT_URI
    )

@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        if not user_info:
            raise AppException(
                status_code=401,
                message="Unable to retrieve Google user information",
            )


        return auth_service.google_login(
                user_info=dict(user_info),
                db=db,
            )
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Google callback failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Google callback failed')
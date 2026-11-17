
from app.schemas.auth_schema import RefreshTokenResponseSchema
import logging
from app.core.exception import AppException
from app.core.config import settings
from app.core.oauth import oauth
from app.schemas.auth_schema import RegisterRequestSchema
from fastapi import APIRouter, Depends, Request, Response
from app.db.base import get_db
from app.services import auth_service
from sqlalchemy.orm import Session
from app.schemas.base_schema import DataResponse
from app.schemas.auth_schema import AuthResponseSchema, LoginRequestSchema

logger = logging.getLogger("app.api.v1.endpoints.auth")

router = APIRouter()

@router.post("/login", description="Login and get access token", response_model=DataResponse[AuthResponseSchema])
async def login(data: LoginRequestSchema, response: Response, db: Session = Depends(get_db)):
    return auth_service.login(data, response, db)

@router.post("/register", description="Register new user", response_model=DataResponse[AuthResponseSchema])
async def register(data: RegisterRequestSchema, response: Response, db: Session = Depends(get_db)):
    return auth_service.register(data, response, db)

@router.get("/google/login")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(
        request, 
        settings.GOOGLE_REDIRECT_URI
    )

@router.get("/google/callback")
async def google_callback(request: Request, response: Response, db: Session = Depends(get_db)):
    try:
        print("SESSION IN CALLBACK:", dict(request.session))
        print("QUERY PARAMS:", dict(request.query_params))
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        if not user_info:
            raise AppException(
                status_code=401,
                message="Unable to retrieve Google user information",
            )


        return auth_service.google_login(
                user_info=dict(user_info),
                response=response,
                db=db,
            )
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Google callback failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Google callback failed')

@router.post("/refresh", description="Refresh access token using HttpOnly Cookie", response_model=DataResponse[RefreshTokenResponseSchema])
async def refresh_token(request: Request, response: Response, db: Session = Depends(get_db)):
    return auth_service.refresh_access_token(request, response, db)
    
@router.post("/logout", description="Logout and clear HttpOnly Cookie", response_model=DataResponse[None])
async def logout(response: Response):
    return auth_service.logout(response)
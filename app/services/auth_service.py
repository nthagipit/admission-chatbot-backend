import logging
from app.enums.user_enum import AuthProviderEnum
from app.core.config import settings
from app.schemas.base_schema import DataResponse
from app.schemas.auth_schema import AuthResponseSchema, RegisterRequestSchema
from app.core.security import create_access_token, create_refresh_token, hash_password, verify_password
from app.core.exception import AppException
from app.models.user_model import User
from sqlalchemy.orm.session import Session
from app.schemas.auth_schema import LoginRequestSchema


logger = logging.getLogger("app.services.auth_service")

def login (data: LoginRequestSchema, db: Session):
    try:
        user = db.query(User).filter(User.email == data.email).first()

        if not user or not user.password_hash:
            raise AppException(
                status_code=401,
                message="Incorrect email or password",
            )

        if not verify_password(data.password, user.password_hash):
            raise AppException(
                status_code=401,
                message="Incorrect email or password",
            )
    except AppException:
        raise
    except Exception as e:
        logger.error("Login failed: %s", str(e), exc_info=True)
        raise AppException(
            status_code=500,
            message="Internal server error",
        )

    return build_auth_response(user)

def register(data: RegisterRequestSchema, db: Session):
    try:
        user = db.query(User).filter(User.email == data.email).first()

        if user:
            logger.warning(f"Failed register attempt for email: {data.email}")
            raise AppException(status_code=400, message='Email already exists')
        
        hashed_password = hash_password(data.password)

        new_user = User(
            full_name=data.full_name,
            email=data.email,
            password_hash=hashed_password,
            auth_provider = AuthProviderEnum.LOCAL
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return build_auth_response(new_user)
    except AppException:
        db.rollback()
        raise
    
    except Exception as e:
        db.rollback()
        logger.error(f"Register failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Register failed')

def build_auth_response(user: User):
    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)

    data_response = AuthResponseSchema(
        id=str(user.id),
        email=user.email,
        access_token=access_token,
        expires_access_token=settings.ACCESS_TOKEN_EXPIRE,
        refresh_token=refresh_token,
        expires_refresh_token=settings.REFRESH_TOKEN_EXPIRE,
        token_type="Bearer",
    )

    return DataResponse.custom_response(
        "200",
        data=data_response,
        message="Login successfully",
    )

def google_login(user_info: dict, db: Session):
    try:    
        if not user_info.get("email_verified"):
            raise AppException (status_code = 401, message = "Google email has not been verified")  
        
        email = user_info.get("email")
        provider_id = user_info.get("sub")
        
        if not email or not provider_id:
            raise AppException(status_code = 400, message = "Invalid Google user info")
        
        user = db.query(User).filter(
            User.auth_provider == AuthProviderEnum.GOOGLE,
            User.provider_id == provider_id
        ).first()

        if not user:
            existing_user = db.query(User).filter(User.email == email).first()
            if existing_user:
                raise AppException(status_code = 400, message = "Email already registered with another login method")
            
            user = User(
                email=email,
                provider_id=provider_id,
                auth_provider=AuthProviderEnum.GOOGLE,
                full_name=user_info.get("name", ""),
                avatar_url=user_info.get("picture", ""),
                password_hash=None,
            )

            db.add(user)
            db.commit()
            db.refresh(user)
        
        return build_auth_response(user)
    except AppException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Google login failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Google login failed')
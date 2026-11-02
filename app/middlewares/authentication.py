from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, status
from sqlalchemy.orm import Session
from app.core.exception import AppException
from app.models.user_model import User
from app.db.base import get_db
from jose import jwt, JWTError
from app.core.config import settings

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials=Depends(security), db: Session=Depends(get_db)) ->User:
    token = credentials.credentials

    credentials_exception = AppException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        message="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=settings.ALGORITHM)
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")

        if user_id is None or token_type != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise AppException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def require_roles(*roles: str):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in roles:
            raise AppException(
                status_code=status.HTTP_403_FORBIDDEN,
                message="You do not have permission to perform this action",
            )
        return current_user
    return role_checker
    

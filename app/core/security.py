from typing import Optional
from app.core.config import settings
from datetime import timedelta, datetime
from passlib.context import CryptContext
from jose import jwt, JWTError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) ->str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)

def create_access_token(user_id: str, role: str, expires_delta: Optional[timedelta] = None):
    if expires_delta:
        expires_delta = datetime.now() + expires_delta
    else:
        expires_delta = datetime.now() + timedelta(minutes=15)

    to_encode = {
        "sub": str(user_id), 
        "role": role,
        "exp": expires_delta, 
        "type": "access"
    }

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(user_id: str, expires_delta: Optional[timedelta] = None):
    if expires_delta:
        expires_delta = datetime.now() + expires_delta
    else:
        expires_delta = datetime.now() + timedelta(days=7)

    to_encode = {
        "sub": str(user_id), 
        "exp": expires_delta, 
        "type": "refresh"
    }

    return jwt.encode(to_encode, settings.REFRESH_SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_refresh_token(token: str):
    try:
        payload = jwt.decode(token, settings.REFRESH_SECRET_KEY, algorithms=[settings.ALGORITHM])

        if payload.get("type") != "refresh":
            return None
        
        return payload
    
    except JWTError :
        return None
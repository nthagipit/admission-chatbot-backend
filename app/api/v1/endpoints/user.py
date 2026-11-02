
from app.db.base import get_db
from sqlalchemy.orm import Session
from app.services import user_service
from app.enums.user_enum import RoleEnum
from app.middlewares.authentication import require_roles
from app.schemas.user_schema import UserSchema
from app.schemas.base_schema import DataResponse
from fastapi import Depends, APIRouter
from app.models.user_model import User
from app.middlewares.authentication import get_current_user


router = APIRouter()

@router.get("/me", description="Get current user", response_model=DataResponse[UserSchema])
async def get_current_user(current_user: User = Depends(get_current_user)):
    return DataResponse.custom_response("200", data=current_user, message="Get current user successfully")

@router.get("", description="Get all users", response_model=DataResponse[list[UserSchema]])
async def get_users(current_user: User = Depends(require_roles(RoleEnum.ADMIN)), db: Session = Depends(get_db)):
    users = user_service.get_all_user(db)
    return DataResponse.custom_response("200",data=users, message="Users retrieved successfully")
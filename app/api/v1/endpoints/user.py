
from fastapi import Query
from app.schemas.base_schema import PageResponse
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


@router.get("/{user_id}", description="Get user by id", response_model=DataResponse[UserSchema])
async def get_user_by_id( 
    user_id: str, 
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    user = user_service.get_current_user(user_id,db)
    return DataResponse.custom_response("200", data=user, message="Get user by id successfully")



@router.get("", description="Get all users", response_model=DataResponse[PageResponse[UserSchema]])
async def get_users(
    page: int = Query(1, ge=1, description="Current page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of records per page"),
    search: str | None = Query(None, description="Search name or email"),
    role: str | None = Query(None, description="Search role"),
    provider: str | None = Query(None, description="Search provider"),
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    response = user_service.get_all_user(page, page_size, search, role, provider, db)
    return DataResponse.custom_response("200",data=response, message="Users retrieved successfully")
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.db.base import get_db
from app.models.user_model import User
from app.middlewares.authentication import require_roles
from app.enums.user_enum import RoleEnum
from app.schemas.base_schema import DataResponse
from app.schemas.dashboard_schema import DashboardResponseSchema
from app.services import admin_service

router = APIRouter()

@router.get(
    "/dashboard",
    description="Lấy toàn bộ số liệu thống kê Dashboard Admin",
    response_model=DataResponse[DashboardResponseSchema]
)
async def get_dashboard(
    days: int = Query(14, ge=1, le=90, description="Số ngày gần nhất cần thống kê biểu đồ"),
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    dashboard_data = admin_service.get_dashboard_data(days=days, db=db)
    return DataResponse.custom_response(
        code="200",
        message="Get dashboard data successfully",
        data=dashboard_data
    )
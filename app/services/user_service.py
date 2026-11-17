from sqlalchemy import or_
from app.utils.pagination import paginate
from app.schemas.base_schema import PageResponse
from app.core.exception import AppException
from app.models.user_model import User
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger("app.services.user_service")

def get_current_user(user_id: int, db: Session):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise AppException(status_code=404, message="User not found")
        return user
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Get user failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Get user failed')

def get_all_user(
    page: int, 
    page_size: int, 
    search: str | None,
    role: str | None,
    provider: str| None,
    db: Session
):
    try:
        query = db.query(User).order_by(User.created_at.desc())
        if search: 
            search_kw = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    User.full_name.ilike(search_kw),
                    User.email.ilike(search_kw)
                )
            )
        
        if role:
            query = query.filter(User.role == role)

        if provider:
            query = query.filter(User.auth_provider == provider)

        return paginate(query=query, page=page, page_size=page_size)
    except Exception as e:
        logger.error(f"Get users failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Get users failed')

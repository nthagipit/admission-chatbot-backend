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

def get_all_user(db: Session):
    try:
        users = db.query(User).all()
        return users
    except Exception as e:
        logger.error(f"Get users failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Get users failed')
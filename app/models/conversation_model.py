
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, func
from app.models.base_model import BaseModel
from datetime import datetime

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.user_model import User
    from app.models.message_model import Message


class Conversation(BaseModel):
    __tablename__ = 'conversations'
    
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete='CASCADE'),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    user: Mapped["User"] = relationship(back_populates = 'conversations')
    messages: Mapped[list["Message"]] = relationship(
        back_populates = 'conversation',
        cascade = "all, delete-orphan",
    )
    
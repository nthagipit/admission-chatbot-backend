from typing import TYPE_CHECKING
from sqlalchemy import String, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base_model import BaseModel
from sqlalchemy.dialects.postgresql import JSONB

if TYPE_CHECKING:
    from app.models.conversation_model import Conversation


class Message(BaseModel):
    __tablename__ = 'messages'
    
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[JSONB] = mapped_column(JSONB, nullable=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False
    )

    conversation: Mapped['Conversation'] = relationship(back_populates = 'messages') 
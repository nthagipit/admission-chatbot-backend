from app.models.base_model import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.conversation_model import Conversation
    from app.models.knowledge_document_model import KnowledgeDocument


class User(BaseModel):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    avatar_url: Mapped[str] = mapped_column(String(255), nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(50), nullable=False, default="local")
    provider_id: Mapped[str] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates = 'user',
        cascade = "all, delete-orphan",

    )

    knowledge_documents: Mapped[list["KnowledgeDocument"]] = relationship(
        back_populates = 'user',
        cascade = "all, delete-orphan",
    )

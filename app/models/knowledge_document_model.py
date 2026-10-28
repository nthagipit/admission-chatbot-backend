from typing import TYPE_CHECKING
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import mapped_column,relationship, Mapped
from app.models.base_model import BaseModel

if TYPE_CHECKING:
    from app.models.user_model import User
    from app.models.document_chunk_model import DocumentChunk

class KnowledgeDocument(BaseModel):
    __tablename__ = 'knowledge_documents'
    
    uploaded_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete='CASCADE'), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    user: Mapped["User"] = relationship(back_populates = 'knowledge_documents')

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates = 'document',
        cascade = "all, delete-orphan",
    )
    
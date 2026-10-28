from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import ForeignKey,Integer,Text, UniqueConstraint
from sqlalchemy.orm import mapped_column, Mapped, relationship
from app.models.base_model import BaseModel
from app.models.knowledge_document_model import KnowledgeDocument

class DocumentChunk(BaseModel):
    __tablename__ = 'document_chunks'
    
    document_id : Mapped[str] = mapped_column(ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index : Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column("metadata",JSONB, nullable=False)
    document: Mapped["KnowledgeDocument"] = relationship(back_populates = 'chunks')

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentResponseSchema(BaseModel):
    id: str
    title: str
    file_name: str
    file_type: str = "PDF"
    file_size: int | None = None
    status: str
    enabled: bool = True
    total_pages: int | None = None
    total_chunks: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DocumentStatsSchema(BaseModel):
    total: int
    completed: int
    processing: int
    failed: int

class DocumentListResponseSchema(BaseModel):
    stats: DocumentStatsSchema
    items: list[DocumentResponseSchema]
    total: int
    page: int
    page_size: int
    total_pages: int
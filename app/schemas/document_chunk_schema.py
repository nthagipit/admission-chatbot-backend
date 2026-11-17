from pydantic import ConfigDict
from datetime import datetime
from pydantic import BaseModel

class DocumentChunkItemSchema(BaseModel):
    id: str
    index: int
    page_number: int | None = None
    section: str | None = None
    content: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class DocumentChunksPageSchema(BaseModel):
    items: list[DocumentChunkItemSchema]
    total: int
    page: int
    page_size: int
    total_pages: int

class DocumentUpdateSchema(BaseModel):
    enabled: bool
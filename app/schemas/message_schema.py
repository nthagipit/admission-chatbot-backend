from datetime import datetime
from pydantic import BaseModel, ConfigDict

class MessageSchema(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict | str] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
    
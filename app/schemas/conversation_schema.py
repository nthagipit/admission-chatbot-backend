from app.schemas.message_schema import MessageSchema
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ConversationSchema(BaseModel):
    id: str
    title:str | None = None
    created_at:datetime
    updated_at:datetime
    
    model_config = ConfigDict(from_attributes=True)

class ConversationMessagesSchema(BaseModel):
   id: str
   title: str | None = None
   messages:list[MessageSchema]

   model_config = ConfigDict(from_attributes=True)
    

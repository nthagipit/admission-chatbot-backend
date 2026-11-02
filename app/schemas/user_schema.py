from datetime import datetime
from pydantic import BaseModel, ConfigDict


class UserSchema(BaseModel):
    id: str
    email: str
    full_name: str
    avatar_url:str | None = None
    auth_provider:str
    provider_id:str | None = None
    role: str
    created_at:datetime
    updated_at:datetime
    
    model_config = ConfigDict(from_attributes=True)

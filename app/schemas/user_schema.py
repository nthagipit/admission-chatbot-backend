from pydantic import BaseModel, ConfigDict


class UserSchema(BaseModel):
    id: str
    email: str
    full_name: str
    avatar_url: str
    auth_provider: str
    provider_id: str
    role: str
    
    model_config = ConfigDict(from_attributes=True)


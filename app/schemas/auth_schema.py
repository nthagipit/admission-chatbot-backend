from pydantic import Field
from pydantic import EmailStr
from pydantic import BaseModel


class LoginRequestSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=5, description="Mật khẩu tối thiểu 5 ký tự")

class RegisterRequestSchema(BaseModel):
    full_name:  str = Field(..., min_length=5, max_length=100, description="Họ và tên")
    email: EmailStr
    password: str = Field(..., min_length=6, description="Mật khẩu tối thiểu 6 ký tự")


class AuthResponseSchema(BaseModel):
    access_token: str
    expires_access_token: int
    refresh_token: str
    expires_refresh_token: int
    token_type: str = "Bearer"

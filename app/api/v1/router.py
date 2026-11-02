from fastapi import APIRouter
from app.api.v1.endpoints import auth, user, conversation

api_router = APIRouter()

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    user.router,
    prefix="/users",
    tags=["User"]
)

api_router.include_router(
    conversation.router,
    prefix="/conversations",
    tags=["Conversation"]
)

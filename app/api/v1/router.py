from fastapi import APIRouter
from app.api.v1.endpoints import auth, user, conversation, admin, knowledge_document

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

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["Admin"]
)

api_router.include_router(
    knowledge_document.router,
    prefix="/documents",
    tags=["Knowledge Documents"]
)

from fastapi.responses import StreamingResponse
from app.schemas.conversation_schema import SendMessageSchema
from app.schemas.conversation_schema import AdminConversationDetailSchema
from app.middlewares.authentication import require_roles
from app.enums.user_enum import RoleEnum
from fastapi import Query
from app.schemas.conversation_schema import AdminConversationItemSchema
from app.schemas.base_schema import PageResponse
from app.schemas.message_schema import MessageSchema
from app.schemas.conversation_schema import ConversationSchema, ConversationMessagesSchema
from app.db.base import get_db
from sqlalchemy.orm import Session
from app.services import conversation_service
from app.schemas.base_schema import DataResponse
from fastapi import Depends, APIRouter
from app.models.user_model import User
from app.middlewares.authentication import get_current_user


router = APIRouter()

@router.post("", description="Create a new conversation", response_model=DataResponse[ConversationSchema])
async def create_new_conversation(title:str | None = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_conversation = conversation_service.create_conversation(current_user.id, title, db)
    return DataResponse.custom_response("201", data=new_conversation, message="Create new conversation successfully")

@router.get("", description="Get all conversations", response_model=DataResponse[list[ConversationSchema]])
async def get_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversations = conversation_service.get_all_conversations(current_user.id,db)
    return DataResponse.custom_response("200",data=conversations, message="Conversation retrieved successfully")

@router.put("/{conversation_id}", description="Update conversation", response_model=DataResponse[ConversationSchema])
async def update_conversation(conversation_id: str, title: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = conversation_service.update_conversation(conversation_id, title, db)
    return DataResponse.custom_response("200",data=conversation, message="Conversation updated successfully")

@router.delete("/{conversation_id}", description="Delete conversation", response_model=DataResponse[ConversationSchema])
async def delete_conversation(conversation_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = conversation_service.delete_conversation(conversation_id, db)
    return DataResponse.custom_response("200",data=conversation, message="Conversation deleted successfully")

@router.get("/{conversation_id}/messages", description="Get messages by conversation id", response_model=DataResponse[ConversationMessagesSchema])
async def get_conversation_by_id(conversation_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conversation = conversation_service.get_conversation_by_id(conversation_id, db)
    return DataResponse.custom_response("200",data=conversation, message="Conversation retrieved successfully")

@router.post("/{conversation_id}/messages", description="Response message", response_model=DataResponse[MessageSchema])
async def response_message(query_text:str, conversation_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    assistant_message = conversation_service.response_message(query_text, conversation_id, db)
    return DataResponse.custom_response("200",data=assistant_message, message="Response message successfully")

@router.get("/admin", description="Admin get all conversation of user", response_model=DataResponse[PageResponse[AdminConversationItemSchema]])
async def get_admin_conversation(
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    page_size: int = Query(10, ge=1, le=100, description="Số lượng mỗi trang"),
    search: str | None = Query(None, description="Tìm theo tiêu đề, tên hoặc email"),
    user_id: str | None = Query(None, description="Lọc theo ID người dùng"),
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    response = conversation_service.get_admin_conversations(page=page, page_size=page_size, search=search, user_id=user_id, db=db)

    return DataResponse.custom_response("200", data=response, message="Retrieval all Conversation Successfully")

@router.get("/{conversation_id}/admin", description="Admin get conversation detail and list messages",response_model=DataResponse[AdminConversationDetailSchema])
async def get_admin_conversation_detail(
    conversation_id: str,
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session=Depends(get_db)
):
    response = conversation_service.get_admin_conversation_detail(conversation_id, db)
    return DataResponse.custom_response(code="200", message="Conversation retrieved successfully", data=response)

@router.post("/{conversation_id}/messages/stream", description="User: send question and receive answer  with pattern Real-time Streaming (SSE)"
)
async def stream_message(conversation_id: str, data: SendMessageSchema, current_user: User = Depends(get_current_user)):
    return StreamingResponse(
        conversation_service.stream_response_message(
            query_text=data.query_text,
            conversation_id=conversation_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
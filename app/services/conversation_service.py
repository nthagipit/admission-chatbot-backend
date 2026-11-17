
from app.db.base import SessionLocal
from app.schemas.conversation_schema import AdminConversationDetailSchema
from app.utils.pagination import paginate
from sqlalchemy import or_
from sqlalchemy import func
from app.enums.user_enum import MessageRole
from app.models.message_model import Message
from app.rag.generation.generator import generate_answer, extract_citations, generate_answer_stream
from app.rag.retrieval.retriever import search_similar_chunks
from app.models.conversation_model import Conversation
from app.core.exception import AppException
from app.models.user_model import User
from sqlalchemy.orm import Session, joinedload
import logging
import json

logger = logging.getLogger("app.services.conversation_service")

def create_conversation(user_id: int, title: str, db: Session):
    try:
        conversation = Conversation(
            user_id = user_id,
            title = title
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Create conversation failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Create conversation failed')

def get_all_conversations(user_id: int, db: Session):
    try:
        conversations = db.query(Conversation).filter(Conversation.user_id == user_id).all()
        return conversations
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Get user failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Get user failed')

def get_conversation_by_id(conversation_id: int, db: Session):
    try:
        conversation = (
            db.query(Conversation)
            .options(joinedload(Conversation.messages))
            .filter(Conversation.id == conversation_id)
            .first()
        )
        if not conversation:
            raise AppException(status_code=404, message="Conversation not found")
        return conversation
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Get conversation failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Get conversation failed')

def update_conversation(conversation_id: int, title: str, db: Session):
    try:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            raise AppException(status_code=404, message="Conversation not found")
        conversation.title = title
        db.commit()
        db.refresh(conversation)
        return conversation
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Update conversation failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Update conversation failed')

def delete_conversation(conversation_id: int, db: Session):
    try:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            raise AppException(status_code=404, message="Conversation not found")
        db.delete(conversation)
        db.commit()
        return True
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message='Delete conversation failed')

def response_message(query_text:str, conversation_id: str, db: Session):
    try:
        user_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=query_text
        )

        db.add(user_message)
        db.commit()
        db.refresh(user_message)


        
        logger.info("1. Started retrieving")
        chunks = search_similar_chunks(query_text, db, top_k=4)

        logger.info("2. Calling LLM in generator.py...")
        result = generate_answer(query_text, chunks)

        assistant_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=result["answer"],
            citations=result["citations"]
        )

        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        return assistant_message
            
    except AppException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Response message failed: {str(e)}", exc_info=True)
        raise AppException(
            status_code=500,
            message="Response message failed"
        )

def get_admin_conversations(
    page: int,
    page_size: int,
    search: str| None,
    user_id: str | None,
    db: Session
):

    query = (
        db.query(
            Conversation.id.label("id"),
            Conversation.title.label("title"),
            Conversation.user_id.label("user_id"),
            User.full_name.label("full_name"),
            User.email.label("email"),
            User.avatar_url.label("avatar_url"),
            func.count(Message.id).label("total_messages"),
            Conversation.created_at.label("created_at"),
            Conversation.updated_at.label("updated_at"),
        )
        .join(User, Conversation.user_id == User.id)
        .outerjoin(Message, Conversation.id == Message.conversation_id)
    )

    if user_id:
        query = query.filter(Conversation.user_id == user_id)

    if search:
        search_kw = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Conversation.title.ilike(search_kw),
                User.full_name.ilike(search_kw),
                User.email.ilike(search_kw)
            )
        )
    
    query = query.group_by(Conversation.id, User.id).order_by(Conversation.updated_at.desc())

    return paginate(query=query, page =page, page_size=page_size)

def get_admin_conversation_detail(conversation_id: str, db: Session):
    try:
        conversation = (
            db.query(Conversation)
            .options(
                joinedload(Conversation.messages),
                joinedload(Conversation.user)
            )
            .filter(Conversation.id == conversation_id)
            .first()
        )

        if not conversation:
            raise AppException(status_code=404, message="Không tìm thấy đoạn hội thoại nào")
        
        sorted_messages = sorted(conversation.messages, key=lambda m: m.created_at)

        return AdminConversationDetailSchema(
            id=conversation.id,
            title=conversation.title,
            user_id=conversation.user_id,
            full_name=conversation.user.full_name if conversation.user else "",
            email=conversation.user.email if conversation.user else "",
            avatar_url= conversation.user.avatar_url if conversation.user else None,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=sorted_messages
        )
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Get admin conversation detail failed: {str(e)}", exc_info=True)
        raise AppException(status_code=400, message="Lấy chi tiết hội thoại thất bại")

def stream_response_message(query_text: str, conversation_id: str):
    """
    Generator stream từng câu trả lời theo chuẩn Server-Sent Events (SSE).
    Mở Session DB riêng để đảm bảo an toàn trong suốt vòng đời stream.
    """
    db: Session = SessionLocal()
    try:
        user_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=query_text
        )
        db.add(user_message)
        db.commit()

        chunks = search_similar_chunks(query_text, db, top_k=4)
        citations = extract_citations(chunks)

        # Send event 'citations' for Frontend show origin
        yield f"data: {json.dumps({'type': 'citations', 'data': citations}, ensure_ascii=False)}\n\n"

        # Stream each token from LLM
        full_answer = []
        for token in generate_answer_stream(query_text, chunks):
            full_answer.append(token)
            yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
        
        complete_content = "".join(full_answer).strip()
        assistant_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=complete_content,
            citations=citations
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)

        yield f"data: {json.dumps({'type': 'done', 'data': {'message_id': assistant_message.id}}, ensure_ascii=False)}\n\n"
    except Exception as e:
        db.rollback()
        logger.error(f"Stream response message failed: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': 'Có lỗi xảy ra khi tạo câu trả lời.'}, ensure_ascii=False)}\n\n"
    finally:
        db.close()

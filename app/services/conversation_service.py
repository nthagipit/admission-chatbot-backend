from app.enums.user_enum import MessageRole
from app.models.message_model import Message
from app.rag.generation.generator import generate_answer
from app.rag.retrieval.retriever import search_similar_chunks
from sqlalchemy.orm import joinedload
from app.models.conversation_model import Conversation
from app.core.exception import AppException
from app.models.user_model import User
from sqlalchemy.orm import Session
import logging

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


        
        print("\n1. Đang truy vấn retrieval...")
        chunks = search_similar_chunks(query_text, db, top_k=4)

        print("\n2. Đang gọi LLM trong generator.py...")
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
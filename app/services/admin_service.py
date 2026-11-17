import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.models.message_model import Message
from app.models.conversation_model import Conversation
from app.models.user_model import User
from app.models.knowledge_document_model import KnowledgeDocument

from app.schemas.dashboard_schema import (
    DashboardResponseSchema,
    DashboardStatsSchema,
    ConversationByDaySchema,
    DocumentStatusItemSchema,
    SystemActivityItemSchema,
    RecentActivityItemSchema,
    RecentDocumentItemSchema,
)

logger = logging.getLogger("app.services.auth_service")

def get_dashboard_stats(db: Session):
    total_docs = db.query(KnowledgeDocument).count()

    completed_docs = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.status == "COMPLETED"
    ).count()

    processing_docs = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.status == "PROCESSING"
    ).count()

    failed_docs = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.status == "FAILED"
    ).count()

    total_users = db.query(User).count()
    total_convs = db.query(Conversation).count()

    doc_status_query = (
        db.query(
            KnowledgeDocument.status,
            func.count(KnowledgeDocument.id).label("count")
        )
        .group_by(KnowledgeDocument.status)
        .all()
    )

    document_status = [
        DocumentStatusItemSchema(
            status=row.status,
            count=row.count
        )
        for row in doc_status_query
    ]

    stats = DashboardStatsSchema(
        total_documents=total_docs,
        completed_documents=completed_docs,
        processing_documents=processing_docs,
        failed_documents=failed_docs,
        total_users=total_users,
        total_conversations=total_convs,
    )

    return stats, document_status


def get_dashboard_analytics(days: int, db: Session):
    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    # Conversations by day
    conv_by_day_query = (
        db.query(
            func.to_char(
                Conversation.created_at, "YYYY-MM-DD"
            ).label("date"),
            func.count(Conversation.id).label("count")
        )
        .filter(Conversation.created_at >= start_date)
        .group_by(
            func.to_char(Conversation.created_at, "YYYY-MM-DD")
        )
        .order_by("date")
        .all()
    )

    conversations_by_day = [
        ConversationByDaySchema(
            date=row.date,
            count=row.count
        )
        for row in conv_by_day_query
    ]

    # Queries by day
    queries_by_day = dict(
        db.query(
            func.to_char(Message.created_at, "YYYY-MM-DD"),
            func.count(Message.id)
        )
        .filter(
            Message.created_at >= start_date,
            Message.role == "user"
        )
        .group_by(
            func.to_char(Message.created_at, "YYYY-MM-DD")
        )
        .all()
    )

    # Uploads by day
    uploads_by_day = dict(
        db.query(
            func.to_char(KnowledgeDocument.created_at, "YYYY-MM-DD"),
            func.count(KnowledgeDocument.id)
        )
        .filter(KnowledgeDocument.created_at >= start_date)
        .group_by(
            func.to_char(KnowledgeDocument.created_at, "YYYY-MM-DD")
        )
        .all()
    )

    # Combine system activity
    all_dates = sorted(
        set(queries_by_day.keys()) | set(uploads_by_day.keys())
    )

    system_activity = [
        SystemActivityItemSchema(
            date=date,
            queries=queries_by_day.get(date, 0),
            uploads=uploads_by_day.get(date, 0)
        )
        for date in all_dates
    ]

    return conversations_by_day, system_activity


def get_recent_activities(db: Session, limit: int = 6):
    activities = []

    # Recent registered users
    recent_users = (
        db.query(User)
        .order_by(User.created_at.desc())
        .limit(5)
        .all()
    )

    for user in recent_users:
        activities.append(
            RecentActivityItemSchema(
                id=f"act-user-{user.id}",
                type="USER_REGISTERED",
                message=f"{user.full_name} vừa đăng ký tài khoản mới",
                created_at=user.created_at
            )
        )

    # Recent document events
    recent_docs = (
        db.query(KnowledgeDocument)
        .order_by(KnowledgeDocument.created_at.desc())
        .limit(5)
        .all()
    )

    for doc in recent_docs:
        act_type = (
            f"DOCUMENT_{doc.status}"
            if doc.status in ["COMPLETED", "FAILED"]
            else "DOCUMENT_UPLOADED"
        )

        message = (
            f"Tài liệu “{doc.title}” đã được tải lên"
            if act_type == "DOCUMENT_UPLOADED"
            else f"Tài liệu “{doc.title}” xử lý {doc.status.lower()}"
        )

        activities.append(
            RecentActivityItemSchema(
                id=f"act-doc-{doc.id}",
                type=act_type,
                message=message,
                created_at=doc.updated_at or doc.created_at
            )
        )

    activities.sort(
        key=lambda activity: activity.created_at,
        reverse=True
    )

    return activities[:limit]


def get_recent_documents(db: Session, limit: int = 5):
    latest_docs = (
        db.query(KnowledgeDocument)
        .order_by(KnowledgeDocument.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        RecentDocumentItemSchema(
            id=doc.id,
            name=doc.title,
            file_type="PDF",
            status=doc.status,
            created_at=doc.created_at
        )
        for doc in latest_docs
    ]


# def get_dashboard_data(days: int, db: Session):
#     stats, document_status = get_dashboard_stats(db)

#     conversations_by_day, system_activity = get_dashboard_analytics(
#         days, db
#     )

#     recent_activities = get_recent_activities(db)

#     recent_documents = get_recent_documents(db)

#     return DashboardResponseSchema(
#         stats=stats,
#         conversations_by_day=conversations_by_day,
#         document_status=document_status,
#         system_activity=system_activity,
#         recent_activities=recent_activities,
#         recent_documents=recent_documents,
#     )

def get_dashboard_data(days: int, db: Session):
    result = {}

    try:
        result["stats"], result["document_status"] = get_dashboard_stats(db)
    except Exception:
        db.rollback()
        logger.exception("Failed to load dashboard stats")
        result["stats"] = None
        result["document_status"] = []

    try:
        (
            result["conversations_by_day"],
            result["system_activity"]
        ) = get_dashboard_analytics(days, db)
    except Exception:
        db.rollback()
        logger.exception("Failed to load dashboard analytics")
        result["conversations_by_day"] = []
        result["system_activity"] = []

    try:
        result["recent_activities"] = get_recent_activities(db)
    except Exception:
        db.rollback()
        logger.exception("Failed to load recent activities")
        result["recent_activities"] = []

    try:
        result["recent_documents"] = get_recent_documents(db)
    except Exception:
        db.rollback()
        logger.exception("Failed to load recent documents")
        result["recent_documents"] = []

    return DashboardResponseSchema(**result)
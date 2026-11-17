from datetime import datetime
from pydantic import BaseModel

class DashboardStatsSchema(BaseModel):
    total_documents: int
    completed_documents: int
    processing_documents: int
    failed_documents: int
    total_users: int
    total_conversations: int

class ConversationByDaySchema(BaseModel):
    date: str
    count: int

class DocumentStatusItemSchema(BaseModel):
    status: str
    count: int

class SystemActivityItemSchema(BaseModel):
    date: str
    queries: int
    uploads: int

class RecentActivityItemSchema(BaseModel):
    id: str
    type: str  # DOCUMENT_UPLOADED, DOCUMENT_COMPLETED, DOCUMENT_FAILED, USER_REGISTERED
    message: str
    created_at: datetime

class RecentDocumentItemSchema(BaseModel):
    id: str
    name: str
    file_type: str = "PDF"
    status: str
    created_at: datetime

class DashboardResponseSchema(BaseModel):
    stats: DashboardStatsSchema
    conversations_by_day: list[ConversationByDaySchema]
    document_status: list[DocumentStatusItemSchema]
    system_activity: list[SystemActivityItemSchema]
    recent_activities: list[RecentActivityItemSchema]
    recent_documents: list[RecentDocumentItemSchema]

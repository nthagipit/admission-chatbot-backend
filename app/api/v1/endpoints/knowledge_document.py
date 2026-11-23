from app.db.base import get_db
from sqlalchemy.orm import Session
from app.models.user_model import User
from app.enums.user_enum import RoleEnum
from app.core.exception import AppException
from fastapi.responses import RedirectResponse
from app.schemas.base_schema import DataResponse
from app.services import knowledge_document_service
from app.middlewares.authentication import require_roles
from app.models.knowledge_document_model import KnowledgeDocument
from app.schemas.document_chunk_schema import DocumentChunksPageSchema,DocumentUpdateSchema
from app.schemas.knowledge_document_schema import DocumentListResponseSchema
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks,Query
from app.schemas.knowledge_document_schema import DocumentResponseSchema

router = APIRouter()

@router.post(
    "/upload",
    description="Admin: Upload knowledge document (PDF) on system",
    response_model=DataResponse[DocumentResponseSchema]
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Knowledge document (PDF))  "),
    title: str | None = Form(None, description="Title of knowledge document (optional)"),
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    result = await knowledge_document_service.upload_document(
        file=file,
        title=title,
        current_user=current_user,
        background_tasks=background_tasks,
        db=db
    )
    return DataResponse.custom_response(
        code="201",
        message="Document uploaded successfully and is being processed",
        data=result
    )

@router.get("", description="Admin: get all documents", response_model=DataResponse[DocumentListResponseSchema])
async def get_all_documents(
    page: int = Query(1, ge=1, description="Number of page"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    search: str | None = Query(None, description="Search by titly"),
    status: str | None = Query(None, description="Filter by status: PROCESSING | COMPLETED | FAILED"),
    sort: str | None = Query("newest", description="Sort: newest | oldest"),
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    result = knowledge_document_service.get_all_documents(
        page=page,
        page_size=page_size,
        search=search,
        status=status,
        sort=sort,
        db=db
    )
    return DataResponse.custom_response(
        code="200",
        message="Documents retrieved successfully",
        data=result
    )

@router.get("/{document_id}", description="Admin: Get document detail for polling and detail page", response_model=DataResponse[DocumentResponseSchema]
)
async def get_document_by_id(
    document_id: str,
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    result = knowledge_document_service.get_document_by_id(document_id, db)
    return DataResponse.custom_response(
        code="200",
        message="Document retrieved successfully",
        data=result
    )

@router.get("/{document_id}/chunks", description="Admin: Get all chunk of document", response_model=DataResponse[DocumentChunksPageSchema])
async def get_document_chunks(
    document_id: str,
    page: int = Query(1, ge=1, description="Number of page"),
    page_size: int = Query(8, ge=1, le=100, description="Number of chunk per page"),
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):

    result = knowledge_document_service.get_document_chunks(document_id, page, page_size, db)

    return DataResponse.custom_response(code="200", message="Document chunks retrieved successfully", data=result)

@router.patch("/{document_id}", description="Admin: turn on or off document (enabled: true/false)", response_model=DataResponse[DocumentResponseSchema])
async def update_document_enabled(
    document_id: str,
    data: DocumentUpdateSchema,
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    result = knowledge_document_service.update_document_enabled(document_id, data, db)
    return DataResponse.custom_response(code="200", message="Document status updated successfully", data=result)


@router.delete("/{document_id}", description="Admin: delete document and vector db in database", response_model=DataResponse[None])
async def delete_document(
    document_id: str,
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    knowledge_document_service.delete_document(document_id, db)
    return DataResponse.custom_response(code="200", message="Document deleted successfully", data=None)

@router.post("/{document_id}/reprocess",description="Admin: reprocess chunking and embedding",response_model=DataResponse[DocumentResponseSchema])
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles(RoleEnum.ADMIN)),
    db: Session = Depends(get_db)
):
    result = knowledge_document_service.reprocess_document(document_id, background_tasks, db)
    return DataResponse.custom_response(code="200", message="Document is being reprocessed", data=result)

@router.get("/{document_id}/view", description="Public/User: Xem trực tiếp file PDF của tài liệu tuyển sinh")
def view_document_pdf(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.id == document_id,
        KnowledgeDocument.enabled == True
    ).first()

    if not doc or not doc.file_path:
        raise AppException(status_code=404, message="Tài liệu không tồn tại hoặc đã bị ẩn")
    
    return RedirectResponse(url=doc.file_path, status_code=307)
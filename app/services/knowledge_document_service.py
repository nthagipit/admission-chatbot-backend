from app.utils.storage import download_file_from_supabase
from app.schemas.document_chunk_schema import DocumentUpdateSchema
from app.schemas.document_chunk_schema import DocumentChunkItemSchema, DocumentChunksPageSchema
from app.schemas.knowledge_document_schema import DocumentListResponseSchema
from sqlalchemy import func
from app.enums.status_document_enum import StatusDocumentEnum
import logging
import os
import uuid
import tempfile
from app.schemas.knowledge_document_schema import DocumentResponseSchema
from app.utils.storage import upload_file_to_supabase, delete_file_from_supabase
from app.core.exception import AppException
from fastapi import BackgroundTasks
from app.models.user_model import User
from app.models.document_chunk_model import DocumentChunk
from app.rag.ingestion.embedding import embedder
from app.rag.ingestion.chunking import chunk_documents
from app.rag.ingestion.chunking import blocks_to_docs
from app.rag.ingestion.pdf_extractor import extract_pdf
from app.models import KnowledgeDocument
from app.db.base import SessionLocal
from sqlalchemy.orm import Session
from fastapi import UploadFile
from app.utils.pagination import paginate

logger = logging.getLogger("app.services.knowledge_document_service")


def check_existing_document(document_id: str, db: Session) -> KnowledgeDocument:
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()

    if not doc:
        logger.error(f"Not found knowledge document {document_id}")
        raise AppException(status_code=404, message="Tài liệu không tồn tại")

    return doc
    

def process_knowledge_document_ingestion(document_id: str, temp_pdf_path: str):
    """
    Hàm này chạy NGẦM (Background Task):
    1. Trích xuất text từ file PDF
    2. Cắt chunk
    3. Tạo embedding vector
    4. Lưu vào PostgreSQL/pgvector
    5. Cập nhật trạng thái COMPLETED hoặc FAILED
    """
    db: Session = SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
        if not doc:
            logger.error(f"Background Ingestion: not found knowledge document {document_id}")
            return
        
        logger.info(f"Starting process RAG for knowledge document: {doc.title}")

        # 1. Extract text & metadata
        extraction_result = extract_pdf(temp_pdf_path)
        blocks = extraction_result.get("blocks",[])
        total_pages = extraction_result.get("total_pages", 0)

        # 2. Chunk
        docs = blocks_to_docs(blocks)
        chunks = chunk_documents(docs)

        if not chunks:
            raise ValueError("No chunk is created from this PDF.")

        # 3. Create vector embeddings
        texts = [chunk["content"] for chunk in chunks]
        vectors = embedder.embed_documents(texts)

        # 4. Insert chunk into DB (remove all old document first)
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()

        chunk_objects = []
        for chunk_data, vector in zip(chunks, vectors):
            chunk_objects.append(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    chunk_metadata=chunk_data["metadata"],
                    embedding=vector
                )
            )
        
            
        db.add_all(chunk_objects)

        # 5. Update status COMPLETED
        doc.status = "COMPLETED"
        doc.total_pages = total_pages
        doc.error_message = None
        db.commit()

        logger.info(f"Finished Ingestion '{doc.title}': {len(chunk_objects)} chunks saved.")
    except Exception as e:
        db.rollback()
        logger.error(f"Ingestion failed for doc {document_id}: {str(e)}", exc_info=True)
        try:
            db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).update({
                "status": StatusDocumentEnum.FAILED,
                "error_message": str(e)
            })
            db.commit()
        except Exception:
            db.rollback()
    
    finally:
        db.close()
        # Clear temporary file on disk
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass

async def upload_document(
    file: UploadFile,
    title: str| None,
    current_user: User,
    background_tasks: BackgroundTasks,
    db: Session
):

    # 1. Check type of PDF
    if not file.filename.lower().endswith(".pdf"):
        raise AppException(status_code=400, message="Hệ thống chỉ chấp nhận định dạng file .pdf")
    
    # 2. Read content file
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise AppException(status_code=400, message="File tải lên không có nội dung")

    doc_title = title.strip() if title and title.strip() else file.name

    # 3. Check override the old document
    existing_doc = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.title == doc_title
    ).first()

    if existing_doc and existing_doc.status == StatusDocumentEnum.PROCESSING:
        raise AppException(
            status_code=400, 
            message=f"Tài liệu '{doc_title}' đang được xử lý, vui lòng chờ hoàn thành trước khi tải đè."
        )

    # 4. Upload to Supabase Storage
    unique_filename = f"{uuid.uuid4().hex[:10]}_{file.filename}"
    public_url = upload_file_to_supabase(
        file_bytes=file_bytes,
        file_name=unique_filename,
        content_type="application/pdf"
    )

    # 5. Saving temporary to disk that pdfplumber can read in background task
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, unique_filename)
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)

    # 6. Create record KnowledgeDocument(status = PROCESSING)
    if existing_doc:
        # Delete the old file on Supabase 
        old_file_name = existing_doc.file_path.split("/")[-1]
        delete_file_from_supabase(old_file_name)
        # Upda old record
        existing_doc.file_path = public_url
        existing_doc.file_size = file_size
        existing_doc.status = StatusDocumentEnum.PROCESSING
        existing_doc.total_pages = None
        existing_doc.error_message = None
        existing_doc.uploaded_by = current_user.id
        db.commit()
        db.refresh(existing_doc)
        target_doc = existing_doc
        logger.info(f"Đang ghi đè tài liệu cũ: {doc_title}")
    else:
        # Create a new record 
        new_doc = KnowledgeDocument(
            title=doc_title,
            file_path=public_url,
            uploaded_by=current_user.id,
            status=StatusDocumentEnum.PROCESSING,
            enabled=True,
            file_size=file_size
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        target_doc = new_doc

    # 7. Throwing task process RAG in under running
    background_tasks.add_task(process_knowledge_document_ingestion, target_doc.id, temp_file_path)

    return DocumentResponseSchema(
        id=target_doc.id,
        title=target_doc.title,
        file_name=file.filename,
        file_type="PDF",
        file_size=target_doc.file_size,
        status=target_doc.status,
        enabled=target_doc.enabled,
        total_pages=None,
        total_chunks=None,
        error_message=None,
        created_at=target_doc.created_at,
        updated_at=target_doc.updated_at,
    )

def get_all_documents(
    page: int,
    page_size: int,
    search: str | None,
    status: str| None,
    sort: str | None,
    db: Session
):
    # 1. Status 4 field
    status_counts = dict(
    db.query(
        KnowledgeDocument.status, 
        func.count(KnowledgeDocument.id)
    )
    .group_by(KnowledgeDocument.status)
    .all()
    )
    completed_count = status_counts.get(StatusDocumentEnum.COMPLETED, 0)
    processing_count = status_counts.get(StatusDocumentEnum.PROCESSING, 0)
    failed_count = status_counts.get(StatusDocumentEnum.FAILED, 0)
    total_all = completed_count + processing_count + failed_count

    stats = {
        "total": total_all,
        "completed": completed_count,
        "processing": processing_count,
        "failed": failed_count
    }

    # 2. Build query list document + count the number of chunk
    query = (
        db.query(
            KnowledgeDocument,
            func.count(DocumentChunk.id).label("total_chunks")
        )
        .outerjoin(DocumentChunk, KnowledgeDocument.id == DocumentChunk.document_id)
        .group_by(KnowledgeDocument.id)
    )
    if search and search.strip():
        search_kw = f"%{search.strip()}%"
        query = query.filter(KnowledgeDocument.title.ilike(search_kw))
    if status and status.strip():
        query = query.filter(KnowledgeDocument.status == status.strip().upper())
    if sort == "oldest":
        query = query.order_by(KnowledgeDocument.created_at.asc())
    else:
        query = query.order_by(KnowledgeDocument.created_at.desc())
    # 3. Panigate for project
    paginated = paginate(query=query, page=page, page_size=page_size)

    # 4. Map items data to DocumentResponseSchema

    items = []
    for doc, total_chunks in paginated["items"]:
        file_name = doc.file_path.split("/")[-1] if doc.file_path else doc.title
        if "_" in file_name:
            file_name = file_name.split("_", 1)[-1]
        items.append(
            DocumentResponseSchema(
                id=doc.id,
                title=doc.title,
                file_name=file_name,
                file_type="PDF",
                file_size=doc.file_size,
                status=doc.status,
                enabled=doc.enabled,
                total_pages=doc.total_pages,
                total_chunks=total_chunks,
                error_message=doc.error_message,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
        )
    
    return DocumentListResponseSchema(
        stats=stats,
        items=items,
        total=paginated["total"],
        page=paginated["page"],
        page_size=paginated["page_size"],
        total_pages=paginated["total_pages"]
    )

def get_document_by_id(document_id: str, db: Session):
    doc = check_existing_document(document_id, db)
    total_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).count()
    file_name = doc.file_path.split("/")[-1] if doc.file_path else doc.title

    if "_" in file_name:
        file_name = file_name.split("_", 1)[-1]

    return DocumentResponseSchema(
        id=doc.id,
        title=doc.title,
        file_name=file_name,
        file_type="PDF",
        file_size=doc.file_size,
        status=doc.status,
        enabled=doc.enabled,
        total_pages=doc.total_pages,
        total_chunks=total_chunks,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )



def get_document_chunks(document_id: str, page: int, page_size: int, db: Session):
    
    check_existing_document(document_id, db)
    
    query = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )

    paginated = paginate(query=query, page=page, page_size=page_size)

    items = []

    for chunk in paginated["items"]:
        meta = chunk.chunk_metadata or {}
        page_num = meta.get("page") or meta.get("page_number")
        section = meta.get("section")

        items.append(
            DocumentChunkItemSchema(
                id=chunk.id,
                index=chunk.chunk_index,
                page_number=page_num,
                section=section,
                content=chunk.content,
                created_at=chunk.created_at
            )
        )

    return DocumentChunksPageSchema(
        items=items,
        total=paginated["total"],
        page=paginated["page"],
        page_size=paginated["page_size"],
        total_pages=paginated["total_pages"]
    )

def update_document_enabled(document_id: str, data: DocumentUpdateSchema, db: Session):
    doc = check_existing_document(document_id, db)

    doc.enabled = data.enabled
    db.commit()
    db.refresh(doc)

    return get_document_by_id(doc.id, db)

def delete_document(document_id: str, db: Session):
    doc = check_existing_document(document_id, db)

    if doc.file_path:
        file_name = doc.file_path.split("/")[-1]
        delete_file_from_supabase(file_name)

    db.delete(doc)
    db.commit()
    return True

def reprocess_document(document_id: str, background_tasks: BackgroundTasks, db: Session):
    doc = check_existing_document(document_id, db)

    if doc.status == StatusDocumentEnum.PROCESSING:
         raise AppException(status_code=400, message="Tài liệu đang trong tiến trình xử lý, không thể chạy lại.")
    
    file_name = doc.file_path.split("/")[-1]
    file_bytes = download_file_from_supabase(file_name)

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file_name)
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)

    doc.status = StatusDocumentEnum.PROCESSING
    doc.error_message = None
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_knowledge_document_ingestion, doc.id, temp_file_path)
    return get_document_by_id(doc.id, db)


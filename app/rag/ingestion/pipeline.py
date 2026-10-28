
from app.rag.ingestion.embedding import embedder
from app.models.document_chunk_model import DocumentChunk
from app.rag.ingestion.chunking import chunk_documents, blocks_to_docs
from app.rag.ingestion.pdf_extractor import extract_pdf
from app.models.knowledge_document_model import KnowledgeDocument
from app.core.security import hash_password
from app.models.user_model import User
from sqlalchemy.orm import Session
from app.db.base import get_db


def get_or_create_admin(db: Session) -> User:
    admin = db.query(User).filter(User.email == "admin@ute.udn.vn").first()

    if not admin:
        admin = User(
            email="admin@ute.udn.vn",
            full_name="Admin",
            password_hash=hash_password("admin"),
            auth_provider="local",
            role="admin",
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)

    return admin


def ingest_pdf_document(file_path: str, title: str, db: Session) -> KnowledgeDocument:
    admin = get_or_create_admin(db)

    doc = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.file_path == file_path
    ).first()

    if doc:
        print(f"Document '{title}' already in database, proceeding to update.")
        doc.title = title
        doc.status = "PROCESSING"
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
    else:
        doc = KnowledgeDocument(
            title=title,
            file_path=file_path,
            uploaded_by=admin.id,
            status="PROCESSING",
        )
        db.add(doc)

    db.commit()
    db.refresh(doc)
    # Cache doc_id trước khi có thể bị detached instance sau rollback
    doc_id = doc.id

    try:
        # ── Bước 1: Extraction (luồng mới) ─────────────────────────
        print("Extracting text from PDF...")
        extraction_result = extract_pdf(file_path)
        # extract_pdf() trả về dict: {"source_file", "total_pages", "blocks": list[dict]}
        blocks = extraction_result["blocks"]
        print(
            f"Extracted {len(blocks)} blocks from "
            f"{extraction_result['total_pages']} pages"
        )

        # ── Bước 2: Chunking (blocks → docs → chunks) ──────────────
        docs = blocks_to_docs(blocks)
        chunks = chunk_documents(docs)
        print(f"Generated {len(chunks)} chunks")

        if not chunks:
            raise ValueError(
                "Pipeline sinh ra 0 chunks từ tài liệu này. "
                "Kiểm tra lại PDF hoặc logic chunking trước khi đánh dấu COMPLETED."
            )

        # ── Bước 3: Validate nội dung chunks ───────────────────────
        texts = [chunk["content"] for chunk in chunks]
        empty_indices = [
            i for i, t in enumerate(texts) if not t or not t.strip()
        ]
        if empty_indices:
            raise ValueError(
                f"Phát hiện {len(empty_indices)} chunk rỗng tại "
                f"index {empty_indices[:5]}. Không thể tiến hành embedding."
            )

        # ── Bước 4: Embedding ───────────────────────────────────────
        print("Embedding texts...")
        vectors = embedder.embed_documents(texts)
        print("Embedding completed")

        # Validate số lượng và số chiều
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Số embedding ({len(vectors)}) không khớp số chunks ({len(chunks)})."
            )
        bad_dim = [i for i, v in enumerate(vectors) if len(v) != 768]
        if bad_dim:
            raise ValueError(
                f"Embedding tại index {bad_dim[:5]} không đúng 768 chiều."
            )

        # ── Bước 5: Insert vào PostgreSQL ───────────────────────────
        chunk_objects = []
        for chunk_data, vector in zip(chunks, vectors):
            chunk_objects.append(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    chunk_metadata=chunk_data["metadata"],
                    embedding=vector,
                )
            )

        db.add_all(chunk_objects)
        doc.status = "COMPLETED"
        db.commit()
        print(f"Successfully ingested '{title}' — {len(chunk_objects)} chunks stored.")
        return doc

    except Exception as exc:
        # Rollback transaction chứa chunk_objects đã add
        db.rollback()

        # Cập nhật FAILED trong một transaction riêng biệt.
        # Dùng query thẳng để tránh lỗi detached instance sau rollback.
        try:
            db.query(KnowledgeDocument).filter(
                KnowledgeDocument.id == doc_id
            ).update({"status": "FAILED"})
            db.commit()
        except Exception as inner_exc:
            db.rollback()
            print(f"[WARN] Không thể cập nhật trạng thái FAILED: {inner_exc}")

        print(f"[ERROR] Ingestion thất bại: {exc}")
        raise exc


if __name__ == "__main__":
    db = next(get_db())

    try:
        file_path = "data/dsk-thong-tin-tuyen-sinh-2026-new-0162026-25-ctdt-1950-ct.pdf"
        title = "Đề án tuyển sinh trình độ đại học năm 2026 - UTE Đà Nẵng"
        ingest_pdf_document(file_path, title, db)
    finally:
        db.close()

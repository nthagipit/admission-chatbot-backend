from app.db.base import SessionLocal
from app.models.document_chunk_model import DocumentChunk

db = SessionLocal()

# Xem chunks index 117-130 (khu vực Bảng 2, 3 và sang trang 29)
chunks = (
    db.query(DocumentChunk)
    .filter(
        DocumentChunk.chunk_index >= 117,
        DocumentChunk.chunk_index <= 135,
    )
    .order_by(DocumentChunk.chunk_index)
    .all()
)

for c in chunks:
    meta = c.chunk_metadata or {}
    print(f"--- chunk_index={c.chunk_index} | Trang {meta.get('page_number')} ---")
    print(c.content)
    print()

db.close()

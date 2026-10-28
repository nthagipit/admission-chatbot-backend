from app.db.base import SessionLocal
from app.models.document_chunk_model import DocumentChunk

db = SessionLocal()

# Tìm chunk chứa 'học phí'
hp_chunks = (
    db.query(DocumentChunk.chunk_index, DocumentChunk.chunk_metadata, DocumentChunk.content)
    .filter(DocumentChunk.content.ilike("%học phí%"))
    .all()
)
print(f"Số chunk chứa 'học phí': {len(hp_chunks)}")
for idx, meta, content in hp_chunks:
    p = (meta or {}).get("page_number")
    first_line = content.split("\n")[0]
    print(f"  - chunk {idx:3d} (trang {p}): {first_line[:70]}")

# Tìm chunk chứa 'công nghệ thông tin'
cntt_chunks = (
    db.query(DocumentChunk.chunk_index, DocumentChunk.chunk_metadata, DocumentChunk.content)
    .filter(DocumentChunk.content.ilike("%công nghệ thông tin%"))
    .all()
)
print(f"\nSố chunk chứa 'công nghệ thông tin': {len(cntt_chunks)}")
for idx, meta, content in cntt_chunks:
    p = (meta or {}).get("page_number")
    first_line = content.split("\n")[0]
    print(f"  - chunk {idx:3d} (trang {p}): {first_line[:70]}")

db.close()

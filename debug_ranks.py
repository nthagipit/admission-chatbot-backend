from app.db.base import SessionLocal
from app.rag.retrieval.retriever import search_similar_chunks, extract_keywords

db = SessionLocal()
query = "Học phí ngành Công nghệ thông tin là bao nhiêu?"
print("Keywords:", extract_keywords(query))

results = search_similar_chunks(query, db, top_k=20)
print(f"\n=== RETRIEVER TOP {len(results)} ===")
for r_idx, r in enumerate(results, start=1):
    print(f"#{r_idx:2d} | chunk_index={r['chunk_index']:3d} | Trang {r['page_number']} | RRF={r['rrf_score']:.5f} | sim={r['similarity']} | {r['content'][:60]}...")

db.close()

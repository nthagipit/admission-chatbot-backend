from app.db.base import SessionLocal
from app.rag.retrieval.retriever import search_similar_chunks
from app.rag.generation.generator import generate_answer

def run_test():
    db = SessionLocal()
    try:
        query = "Trường tuyển bao nhiêu chỉ tiêu ngành Công nghệ thông tin năm 2026?"
        print(f"Câu hỏi test: '{query}'")
        
        print("\n1. Đang truy vấn retrieval...")
        chunks = search_similar_chunks(query, db, top_k=4)
        print(f"-> Lấy được {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n{'=' * 80}")
            print(f"CHUNK #{i}")
            print(f"{'=' * 80}")

            for key, value in chunk.items():
                print(f"{key}: {value}")

        print("\n2. Đang gọi LLM trong generator.py...")
        result = generate_answer(query, chunks)

        print("\n" + "="*30 + " KẾT QUẢ TRẢ LỜI " + "="*30)
        print(result["answer"])
        print("="*77)

        print("\n=== TRÍCH DẪN NGUỒN ===")
        for c in result["citations"]:
            print(f"- {c['document_title']} (Trang {c['page_number']})")
            
    except Exception as e:
        print("\n[LỖI]:", type(e).__name__, "-", e)
    finally:
        db.close()

if __name__ == "__main__":
    run_test()

"""
test_retriever.py — Unit tests for the RAG Retriever module (Hybrid Search & pgvector).

Tests cover:
  1. Question uses embed_query(), not embed_document().
  2. Query uses exact vector returned by embed_query().
  3. Results respect top_k limit.
  4. Chunk content, page_number, source_pages, section and metadata are preserved.
  5. Empty/invalid queries return empty list without executing DB queries.
  6. Empty database (no matching chunks) returns empty list cleanly.
  7. Embedding failure (e.g. 429) propagates exception clearly.
  8. Database query failure propagates exception.
  9. Filtering by document_id and document status ("COMPLETED") is applied.
  10. Reciprocal Rank Fusion (RRF) scoring combines and ranks vector and keyword results properly.
  11. Keyword extraction logic (n-grams, stop words, codes, single word fallback).
"""

import unittest
from unittest.mock import MagicMock, patch, call

from app.rag.retrieval.retriever import search_similar_chunks, extract_keywords
from app.models.document_chunk_model import DocumentChunk
from app.models.knowledge_document_model import KnowledgeDocument


class FakeChunk:
    """Helper mock object mimicking DocumentChunk model."""
    def __init__(self, chunk_id, document_id, content, chunk_index=0, chunk_metadata=None):
        self.id = chunk_id
        self.document_id = document_id
        self.content = content
        self.chunk_index = chunk_index
        self.chunk_metadata = chunk_metadata or {}


class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.mock_db = MagicMock()
        self.fake_vector = [0.05] * 768

    # 1. Câu hỏi sử dụng embed_query(), không sử dụng embed_document()
    @patch("app.rag.retrieval.retriever.embedder")
    def test_01_query_uses_embed_query_not_embed_document(self, mock_embedder):
        """Retriever phải dùng embed_query(query), tuyệt đối không dùng embed_document()."""
        mock_embedder.embed_query.return_value = self.fake_vector

        # Mock DB queries return empty list
        mock_query = MagicMock()
        self.mock_db.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        query_text = "Học phí ngành Công nghệ thông tin là bao nhiêu?"
        search_similar_chunks(query_text, self.mock_db, top_k=4)

        # embed_query được gọi với đúng query_text
        mock_embedder.embed_query.assert_called_once_with(query_text)
        # embed_document KHÔNG được gọi
        mock_embedder.embed_document.assert_not_called()

    # 2. Truy vấn sử dụng đúng vector được tạo từ câu hỏi
    @patch("app.rag.retrieval.retriever.embedder")
    def test_02_query_vector_passed_to_cosine_distance(self, mock_embedder):
        """Vector sinh ra từ embed_query phải được truyền trực tiếp vào biểu thức cosine_distance."""
        test_vec = [0.123] * 768
        mock_embedder.embed_query.return_value = test_vec

        mock_query = MagicMock()
        self.mock_db.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        search_similar_chunks("Chỉ tiêu 2026", self.mock_db, top_k=4)

        # 1. embed_query được gọi với câu hỏi
        mock_embedder.embed_query.assert_called_once_with("Chỉ tiêu 2026")

        # 2. db.query() được gọi với DocumentChunk, KnowledgeDocument.title và distance_col
        self.mock_db.query.assert_called()
        call_args = self.mock_db.query.call_args_list[0][0]
        self.assertEqual(call_args[0], DocumentChunk)
        self.assertEqual(call_args[1], KnowledgeDocument.title)

        # 3. distance_col là Label của BinaryExpression cosine_distance mang đúng test_vec
        distance_label = call_args[2]
        self.assertEqual(distance_label.name, "distance")
        self.assertEqual(distance_label.element.right.value, test_vec)

    # 3. Kết quả tuân thủ top_k
    @patch("app.rag.retrieval.retriever.embedder")
    def test_03_results_respect_top_k(self, mock_embedder):
        """Kết quả trả về không được vượt quá số lượng top_k yêu cầu."""
        mock_embedder.embed_query.return_value = self.fake_vector

        # Tạo 10 candidates giả lập từ vector_results
        fake_vector_results = [
            (
                FakeChunk(f"c_{i}", "doc_1", f"Nội dung chunk {i}", i, {"page": i}),
                "Đề án tuyển sinh 2026",
                0.1 * i,
            )
            for i in range(10)
        ]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_vector_results

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = []

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        # top_k = 3
        res_3 = search_similar_chunks("Học phí", self.mock_db, top_k=3)
        self.assertEqual(len(res_3), 3)

        # top_k = 5
        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]
        res_5 = search_similar_chunks("Học phí", self.mock_db, top_k=5)
        self.assertEqual(len(res_5), 5)

    # 4. Metadata và nội dung chunk được giữ nguyên
    @patch("app.rag.retrieval.retriever.embedder")
    def test_04_metadata_and_content_preserved(self, mock_embedder):
        """Metadata (page, source_pages, section, content_type) và content được bảo toàn."""
        mock_embedder.embed_query.return_value = self.fake_vector

        chunk = FakeChunk(
            chunk_id="chunk_123",
            document_id="doc_ute",
            content="Bảng học phí các ngành kỹ thuật: 18.5 triệu/năm",
            chunk_index=5,
            chunk_metadata={
                "page": 22,
                "source_pages": [21, 22],
                "section": "Mục 4. Học phí",
                "content_type": "table",
                "table_id": "p22_t0",
            },
        )
        fake_vector_results = [(chunk, "Đề án Tuyển sinh UTE Đà Nẵng", 0.15)]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_vector_results

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = []

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks("Học phí ngành kỹ thuật", self.mock_db, top_k=1)
        self.assertEqual(len(results), 1)
        r = results[0]

        self.assertEqual(r["chunk_id"], "chunk_123")
        self.assertEqual(r["document_id"], "doc_ute")
        self.assertEqual(r["content"], "Bảng học phí các ngành kỹ thuật: 18.5 triệu/năm")
        self.assertEqual(r["document_title"], "Đề án Tuyển sinh UTE Đà Nẵng")
        self.assertEqual(r["page_number"], 22)
        self.assertEqual(r["source_pages"], [21, 22])
        self.assertEqual(r["section"], "Mục 4. Học phí")
        self.assertEqual(r["content_type"], "table")
        self.assertEqual(r["chunk_index"], 5)
        self.assertEqual(r["similarity"], 0.85)  # 1 - 0.15 = 0.85
        self.assertGreater(r["rrf_score"], 0)

    # 5. Xử lý câu hỏi rỗng hoặc top_k không hợp lệ
    def test_05_empty_query_or_invalid_top_k(self):
        """Câu hỏi rỗng, chỉ có khoảng trắng, hoặc top_k < 1 phải trả về danh sách rỗng."""
        self.assertEqual(search_similar_chunks("", self.mock_db, top_k=4), [])
        self.assertEqual(search_similar_chunks("   ", self.mock_db, top_k=4), [])
        self.assertEqual(search_similar_chunks(None, self.mock_db, top_k=4), [])
        self.assertEqual(search_similar_chunks("Học phí", self.mock_db, top_k=0), [])
        self.assertEqual(search_similar_chunks("Học phí", self.mock_db, top_k=-1), [])
        # Không có query nào gọi DB
        self.mock_db.query.assert_not_called()

    # 6. Xử lý database rỗng / không tìm được kết quả
    @patch("app.rag.retrieval.retriever.embedder")
    def test_06_empty_database_returns_empty_list(self, mock_embedder):
        """Khi database chưa có embedding hoặc không tìm thấy kết quả nào, trả về [] an toàn."""
        mock_embedder.embed_query.return_value = self.fake_vector

        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        self.mock_db.query.return_value = mock_query

        results = search_similar_chunks("Một câu hỏi không có trong dữ liệu", self.mock_db, top_k=4)
        self.assertEqual(results, [])

    # 7. Xử lý lỗi embedding phù hợp
    @patch("app.rag.retrieval.retriever.embedder")
    def test_07_embedding_failure_propagates_exception(self, mock_embedder):
        """Lỗi từ dịch vụ embedding (429, mạng...) phải được raise rõ ràng, không nuốt lỗi."""
        mock_embedder.embed_query.side_effect = Exception("429 RESOURCE_EXHAUSTED: Gemini rate limit")

        with self.assertRaises(Exception) as ctx:
            search_similar_chunks("Câu hỏi tuyển sinh", self.mock_db, top_k=4)

        self.assertIn("429 RESOURCE_EXHAUSTED", str(ctx.exception))
        self.mock_db.query.assert_not_called()

    # 8. Xử lý lỗi database phù hợp
    @patch("app.rag.retrieval.retriever.embedder")
    def test_08_database_failure_propagates_exception(self, mock_embedder):
        """Lỗi kết nối cơ sở dữ liệu phải được raise rõ ràng."""
        mock_embedder.embed_query.return_value = self.fake_vector
        self.mock_db.query.side_effect = Exception("OperationalError: database connection closed")

        with self.assertRaises(Exception) as ctx:
            search_similar_chunks("Câu hỏi tuyển sinh", self.mock_db, top_k=4)

        self.assertIn("OperationalError", str(ctx.exception))

    # 9. Kiểm tra các điều kiện lọc: document_id và status COMPLETED
    @patch("app.rag.retrieval.retriever.embedder")
    def test_09_filtering_by_document_status_and_id(self, mock_embedder):
        """Truy vấn phải luôn lọc tài liệu COMPLETED và hỗ trợ lọc theo document_id nếu có."""
        mock_embedder.embed_query.return_value = self.fake_vector

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = []

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = []

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        # Gọi với document_id cụ thể
        search_similar_chunks("Phương thức xét tuyển", self.mock_db, top_k=4, document_id="doc_selected_123")

        # filter được gọi ít nhất 2 lần trên vector_query (1 lần cho status COMPLETED, 1 lần cho document_id)
        self.assertGreaterEqual(mock_v_query.filter.call_count, 2)
        self.assertGreaterEqual(mock_k_query.filter.call_count, 2)

    # 10. Reciprocal Rank Fusion (RRF) scoring và ranking
    @patch("app.rag.retrieval.retriever.embedder")
    def test_10_rrf_scoring_and_ranking(self, mock_embedder):
        """RRF score kết hợp từ cả vector search và keyword search; chunk xuất hiện ở cả hai có rank cao nhất."""
        mock_embedder.embed_query.return_value = self.fake_vector

        chunk_a = FakeChunk("chunk_A", "doc_1", "Nội dung A - cả hai đều tìm thấy")
        chunk_b = FakeChunk("chunk_B", "doc_1", "Nội dung B - chỉ vector tìm thấy")
        chunk_c = FakeChunk("chunk_C", "doc_1", "Nội dung C - chỉ keyword tìm thấy")

        # Vector search trả về [Chunk A (rank 1), Chunk B (rank 2)]
        fake_vector_results = [
            (chunk_a, "Doc 1", 0.1),
            (chunk_b, "Doc 1", 0.2),
        ]
        # Keyword search trả về [Chunk A (rank 1), Chunk C (rank 2)]
        fake_keyword_results = [
            (chunk_a, "Doc 1"),
            (chunk_c, "Doc 1"),
        ]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_vector_results

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = fake_keyword_results

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks("Công nghệ thông tin", self.mock_db, top_k=3)
        self.assertEqual(len(results), 3)

        # Chunk A xuất hiện ở rank 1 của cả 2 nguồn -> RRF score = 1/(60+1) + 1/(60+1) ≈ 0.03278
        # Chunk B: 1/(60+2) ≈ 0.01612
        # Chunk C: 1/(60+2) ≈ 0.01612
        self.assertEqual(results[0]["chunk_id"], "chunk_A")
        expected_score_a = 1 / 61 + 1 / 61
        self.assertAlmostEqual(results[0]["rrf_score"], expected_score_a, places=6)

        # Cả chunk B và C đều có mặt
        chunk_ids = {r["chunk_id"] for r in results}
        self.assertEqual(chunk_ids, {"chunk_A", "chunk_B", "chunk_C"})

    # 11. Keyword extraction logic
    def test_11_extract_keywords(self):
        """extract_keywords trích xuất đúng cụm từ khóa, giữ mã số, loại bỏ stop words."""
        # 1. Câu hỏi chuẩn
        kw1 = extract_keywords("Học phí ngành Công nghệ thông tin là bao nhiêu?")
        self.assertIn("công nghệ thông tin", kw1)
        self.assertIn("học phí", kw1)

        # 2. Câu hỏi có số/mã xét tuyển
        kw2 = extract_keywords("Điểm chuẩn ngành Ô tô điện mã 7510205A năm 2026")
        self.assertTrue(any("2026" in k or k == "2026" for k in kw2))

        # 3. Câu hỏi rỗng
        self.assertEqual(extract_keywords(""), [])
        self.assertEqual(extract_keywords("   "), [])
        self.assertEqual(extract_keywords(None), [])

    # 12. _keyword_weight: topic keywords nhận trọng số cao hơn entity keywords
    def test_12_topic_keyword_weight_higher_than_entity(self):
        """Topic keywords (học phí, điểm chuẩn, phương thức) nhận weight cao hơn entity keywords."""
        from app.rag.retrieval.retriever import _keyword_weight, _TOPIC_WEIGHT, _ENTITY_WEIGHT

        # Topic keywords đã biết trong _TOPIC_TERMS
        self.assertEqual(_keyword_weight("học phí"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("điểm chuẩn"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("phương thức xét tuyển"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("xét tuyển"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("học bổng"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("chỉ tiêu"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("tín chỉ"), _TOPIC_WEIGHT)

        # Topic keywords khớp case-insensitive
        self.assertEqual(_keyword_weight("Học Phí"), _TOPIC_WEIGHT)
        self.assertEqual(_keyword_weight("ĐIỂM CHUẨN"), _TOPIC_WEIGHT)

        # Topic weight phải lớn hơn entity weight
        self.assertGreater(_TOPIC_WEIGHT, _ENTITY_WEIGHT)

    # 13. _keyword_weight: entity keywords (tên ngành, mã ngành) nhận trọng số thấp
    def test_13_entity_keyword_weight_is_low(self):
        """Entity keywords (tên ngành, mã ngành) nhận weight thấp hơn topic keywords."""
        from app.rag.retrieval.retriever import _keyword_weight, _ENTITY_WEIGHT

        # Tên ngành không nằm trong _TOPIC_TERMS → weight = _ENTITY_WEIGHT
        self.assertEqual(_keyword_weight("công nghệ thông tin"), _ENTITY_WEIGHT)
        self.assertEqual(_keyword_weight("cơ khí"), _ENTITY_WEIGHT)
        self.assertEqual(_keyword_weight("kỹ thuật điện"), _ENTITY_WEIGHT)
        self.assertEqual(_keyword_weight("7480201"), _ENTITY_WEIGHT)   # mã ngành
        self.assertEqual(_keyword_weight("2026"), _ENTITY_WEIGHT)       # năm

    # 14. RRF: chunk học phí xếp trước chunk chỉ nhắc tên ngành
    @patch("app.rag.retrieval.retriever.embedder")
    def test_14_tuition_chunk_ranks_above_admission_table_after_fix(self, mock_embedder):
        """
        Sau khi sửa weighted keyword scoring:
        - Chunk học phí (khớp topic keyword 'học phí', weight=3) → k_rank cao
        - Chunk xét tuyển (chỉ khớp 'công nghệ thông tin', weight=1) → k_rank thấp hơn
        Kết quả RRF: chunk học phí phải xếp trên chunk xét tuyển.

        Test mô phỏng kết quả sau khi SQL đã order_by weighted keyword_score.
        """
        mock_embedder.embed_query.return_value = [0.1] * 768

        # Chunk A: bảng học phí — v_rank=2 (vector biết đây là câu về học phí)
        tuition_chunk = FakeChunk(
            chunk_id="tuition-001",
            document_id="doc-001",
            content="Học phí năm 2026: 567.100 đồng/tín chỉ cho kỹ sư chưa kiểm định",
            chunk_metadata={"page": 28, "source_pages": [28], "section": "10.1. Thông tin về học phí"},
        )
        # Chunk B: bảng xét tuyển — v_rank=8 (vector ít liên quan)
        admission_chunk = FakeChunk(
            chunk_id="admission-002",
            document_id="doc-001",
            content="Công nghệ thông tin: 75 chỉ tiêu, Phương thức 1; II. Phương thức 2",
            chunk_metadata={"page": 5, "source_pages": [5], "section": "4. Số lượng tuyển sinh"},
        )

        # Vector results: tuition ở v_rank=2 (tương đối thấp), admission ở v_rank=8
        fake_vector_results = [
            (tuition_chunk, "Đề án tuyển sinh 2026", 0.30),   # sim=0.70
            (admission_chunk, "Đề án tuyển sinh 2026", 0.40),  # sim=0.60
        ]

        # Keyword results SAU KHI đã tính weighted score:
        # tuition: matched 'học phí' (weight=3) → keyword_score=3
        # admission: matched 'công nghệ thông tin' (weight=1) → keyword_score=1
        # → tuition được xếp k_rank=1, admission k_rank=2
        fake_keyword_results = [
            (tuition_chunk, "Đề án tuyển sinh 2026"),   # k_rank=1 (score=3)
            (admission_chunk, "Đề án tuyển sinh 2026"), # k_rank=2 (score=1)
        ]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_vector_results

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = fake_keyword_results

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks(
            "Học phí ngành Công nghệ thông tin là bao nhiêu?",
            self.mock_db,
            top_k=2,
        )

        self.assertEqual(len(results), 2)

        # RRF scores:
        # tuition:   1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # admission: 1/(60+2) + 1/(60+2) = 0.01613 + 0.01613 = 0.03226  (v_rank=2)
        # → tuition phải xếp trên admission
        self.assertEqual(
            results[0]["chunk_id"], "tuition-001",
            "Chunk học phí phải xếp rank 1 khi keyword weighted scoring ưu tiên topic term 'học phí'.",
        )
        self.assertEqual(results[1]["chunk_id"], "admission-002")

        # RRF score của tuition phải cao hơn admission
        self.assertGreater(results[0]["rrf_score"], results[1]["rrf_score"])

    # 15. extract_keywords trả về cả topic keyword lẫn entity keyword cho câu hỏi tuyển sinh
    def test_15_keywords_include_both_topic_and_entity(self):
        """
        Câu hỏi tuyển sinh thường có cả topic keyword (điểm chuẩn, học phí, phương thức)
        lẫn entity keyword (tên ngành). Cả hai phải được extract.
        """
        from app.rag.retrieval.retriever import _keyword_weight, _TOPIC_WEIGHT, _ENTITY_WEIGHT

        # Câu hỏi học phí
        kw_tuiton = extract_keywords("Học phí ngành Công nghệ thông tin là bao nhiêu?")
        topic_kws = [kw for kw in kw_tuiton if _keyword_weight(kw) == _TOPIC_WEIGHT]
        entity_kws = [kw for kw in kw_tuiton if _keyword_weight(kw) == _ENTITY_WEIGHT]
        self.assertTrue(len(topic_kws) >= 1, f"Phải có ít nhất 1 topic keyword, got {kw_tuiton}")
        self.assertTrue(len(entity_kws) >= 1, f"Phải có ít nhất 1 entity keyword, got {kw_tuiton}")
        self.assertIn("học phí", topic_kws)
        self.assertIn("công nghệ thông tin", entity_kws)

        # Câu hỏi điểm chuẩn
        kw_score = extract_keywords("Điểm chuẩn ngành Cơ khí năm 2025 là bao nhiêu?")
        topic_score_kws = [kw for kw in kw_score if _keyword_weight(kw) == _TOPIC_WEIGHT]
        self.assertTrue(
            any("điểm chuẩn" in kw for kw in topic_score_kws),
            f"'điểm chuẩn' phải được nhận diện là topic keyword, got {kw_score}",
        )

        # Câu hỏi phương thức xét tuyển
        kw_method = extract_keywords("Phương thức xét tuyển ngành Điện tử viễn thông?")
        topic_method_kws = [kw for kw in kw_method if _keyword_weight(kw) == _TOPIC_WEIGHT]
        self.assertTrue(len(topic_method_kws) >= 1, f"Phải nhận diện topic keyword, got {kw_method}")

    # 16. Nhóm 1: Học phí ngành CNTT — chunk bảng học phí vượt chunk tiêu đề và chunk xét tuyển
    @patch("app.rag.retrieval.retriever.embedder")
    def test_16_tuition_question_reranking(self, mock_embedder):
        """
        Nhóm 1: 'Học phí ngành Công nghệ thông tin là bao nhiêu?'
        - Chunk bảng học phí (có 567.100 đ/tín chỉ, 18.5 triệu) được boost direct answer.
        - Chunk chỉ là tiêu đề '10.1. Thông tin về học phí' bị phạt header_only.
        - Chunk bảng xét tuyển (chỉ nhắc CNTT nhưng không có số tiền) bị phạt topic_mismatch.
        Kết quả: Bảng học phí phải đứng Rank 1, chunk tiêu đề và xét tuyển bị tụt hạng.
        """
        mock_embedder.embed_query.return_value = self.fake_vector

        header_chunk = FakeChunk(
            chunk_id="chunk-heading-01",
            document_id="doc-01",
            content="10.1. Thông tin về học phí",
            chunk_metadata={"content_type": "heading", "page": 28, "section": "10.1. Thông tin về học phí"},
        )
        tuition_table_chunk = FakeChunk(
            chunk_id="chunk-tuition-02",
            document_id="doc-01",
            content="| STT | Khối ngành | Học phí |\n| 1 | Kỹ thuật, Công nghệ thông tin | 567.100 đồng/tín chỉ (18.5 triệu đồng/năm) |",
            chunk_metadata={"content_type": "table", "page": 28, "section": "10.1. Thông tin về học phí"},
        )
        admission_chunk = FakeChunk(
            chunk_id="chunk-admission-03",
            document_id="doc-01",
            content="| Mã ngành | Tên ngành | Chỉ tiêu |\n| 7480201 | Công nghệ thông tin | 75 |",
            chunk_metadata={"content_type": "table", "page": 5, "section": "4. Chỉ tiêu tuyển sinh"},
        )

        # Giả lập: Cả 3 chunks đều lọt vào candidate pool với RRF tương đương nhau
        fake_v = [
            (header_chunk, "Đề án", 0.20),        # sim=0.80
            (tuition_table_chunk, "Đề án", 0.22), # sim=0.78
            (admission_chunk, "Đề án", 0.25),     # sim=0.75
        ]
        fake_k = [
            (admission_chunk, "Đề án"),     # match "công nghệ thông tin"
            (header_chunk, "Đề án"),        # match "học phí"
            (tuition_table_chunk, "Đề án"), # match "học phí"
        ]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_v

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = fake_k

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks(
            "Học phí ngành Công nghệ thông tin là bao nhiêu?",
            self.mock_db,
            top_k=3,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["chunk_id"], "chunk-tuition-02", "Chunk bảng học phí phải xếp Rank 1 sau reranking.")
        self.assertIn("đồng/tín chỉ", results[0]["content"])
        # Chunk header-only phải xếp sau chunk bảng học phí
        chunk_ids = [r["chunk_id"] for r in results]
        self.assertLess(chunk_ids.index("chunk-tuition-02"), chunk_ids.index("chunk-heading-01"))

    # 17. Nhóm 2: Điểm chuẩn của một ngành — chunk điểm chuẩn vượt chunk học phí và phương thức
    @patch("app.rag.retrieval.retriever.embedder")
    def test_17_benchmark_question_reranking(self, mock_embedder):
        """
        Nhóm 2: 'Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?'
        - Chunk bảng điểm chuẩn trúng tuyển phải xếp Rank 1.
        - Chunk học phí và chunk phương thức chung bị phạt do topic_mismatch.
        """
        mock_embedder.embed_query.return_value = self.fake_vector

        benchmark_chunk = FakeChunk(
            chunk_id="chunk-benchmark-01",
            document_id="doc-01",
            content="| Ngành | Mã ngành | Điểm trúng tuyển 2024 | Điểm trúng tuyển 2025 |\n| Công nghệ thông tin | 7480201 | 25.5 | 26.25 |",
            chunk_metadata={"content_type": "table", "page": 35, "section": "10.2. Điểm trúng tuyển"},
        )
        tuition_chunk = FakeChunk(
            chunk_id="chunk-tuition-02",
            document_id="doc-01",
            content="| Ngành | Mức thu |\n| Công nghệ thông tin | 567.100 đồng/tín chỉ |",
            chunk_metadata={"content_type": "table", "page": 28, "section": "10.1. Thông tin về học phí"},
        )

        fake_v = [
            (tuition_chunk, "Đề án", 0.25),
            (benchmark_chunk, "Đề án", 0.26),
        ]
        fake_k = [
            (tuition_chunk, "Đề án"),
            (benchmark_chunk, "Đề án"),
        ]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_v

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = fake_k

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks(
            "Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?",
            self.mock_db,
            top_k=2,
        )

        self.assertEqual(results[0]["chunk_id"], "chunk-benchmark-01")
        self.assertIn("26.25", results[0]["content"])

    # 18. Nhóm 3: Phương thức xét tuyển — chunk mô tả phương thức xếp trên chunk học phí
    @patch("app.rag.retrieval.retriever.embedder")
    def test_18_admission_methods_question_reranking(self, mock_embedder):
        """
        Nhóm 3: 'Phương thức xét tuyển của trường gồm những phương thức nào?'
        - Chunk mô tả các phương thức xét tuyển phải xếp Rank 1.
        - Chunk bảng học phí thuần túy bị phạt topic_mismatch.
        """
        mock_embedder.embed_query.return_value = self.fake_vector

        method_chunk = FakeChunk(
            chunk_id="chunk-method-01",
            document_id="doc-01",
            content="Trường áp dụng 5 phương thức xét tuyển: Phương thức 1 xét tuyển thẳng; Phương thức 2 xét điểm thi THPT; Phương thức 3 xét học bạ THPT; Phương thức 4 xét đánh giá năng lực; Phương thức 5 xét tuyển kết hợp.",
            chunk_metadata={"content_type": "text", "page": 8, "section": "2. Các phương thức xét tuyển"},
        )
        tuition_chunk = FakeChunk(
            chunk_id="chunk-tuition-02",
            document_id="doc-01",
            content="Mức học phí năm 2026 là 567.100 đồng/tín chỉ.",
            chunk_metadata={"content_type": "text", "page": 28, "section": "10.1. Thông tin về học phí"},
        )

        fake_v = [(tuition_chunk, "Đề án", 0.20), (method_chunk, "Đề án", 0.22)]
        fake_k = [(tuition_chunk, "Đề án"), (method_chunk, "Đề án")]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_v

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = fake_k

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks(
            "Phương thức xét tuyển của trường gồm những phương thức nào?",
            self.mock_db,
            top_k=2,
        )

        self.assertEqual(results[0]["chunk_id"], "chunk-method-01")
        self.assertIn("Phương thức 1", results[0]["content"])

    # 19. Nhóm 4: Câu hỏi không có dữ liệu phù hợp (out of domain)
    @patch("app.rag.retrieval.retriever.embedder")
    def test_19_out_of_domain_question_neutral_reranking(self, mock_embedder):
        """
        Nhóm 4: Câu hỏi ngoài tri thức 'Thời tiết tại trường hôm nay như thế nào?'
        - Topic nhận diện là unknown/general.
        - Không tự ý boost điểm direct answer cho chunk không liên quan.
        - Chunk header-only vẫn bị phạt hạ hạng an toàn.
        """
        mock_embedder.embed_query.return_value = self.fake_vector

        header_chunk = FakeChunk(
            chunk_id="chunk-header-99",
            document_id="doc-01",
            content="1. Giới thiệu chung",
            chunk_metadata={"content_type": "heading", "page": 1, "section": "1. Giới thiệu chung"},
        )
        content_chunk = FakeChunk(
            chunk_id="chunk-content-99",
            document_id="doc-01",
            content="Trường Đại học Sư phạm Kỹ thuật là trường công lập trực thuộc Đại học Đà Nẵng, tọa lạc tại số 48 Cao Thắng, Đà Nẵng.",
            chunk_metadata={"content_type": "text", "page": 1, "section": "1. Giới thiệu chung"},
        )

        fake_v = [(header_chunk, "Đề án", 0.50), (content_chunk, "Đề án", 0.55)]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_v

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = []

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks(
            "Thời tiết tại trường hôm nay như thế nào?",
            self.mock_db,
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        # Content chunk phải đứng trước header-only chunk
        self.assertEqual(results[0]["chunk_id"], "chunk-content-99")
        self.assertEqual(results[1]["chunk_id"], "chunk-header-99")

    # 20. Nhóm 5: Chính sách miễn giảm học phí — chunk miễn giảm vượt chunk mức thu học phí
    @patch("app.rag.retrieval.retriever.embedder")
    def test_20_tuition_waiver_policy_question_reranking(self, mock_embedder):
        """
        Nhóm 5: 'Chính sách miễn giảm học phí của trường như thế nào?'
        - Chunk chính sách miễn giảm học phí phải xếp Rank 1 (được boost sub_topic waiver_policy).
        - Chunk bảng mức thu tín chỉ thông thường bị xếp sau.
        """
        mock_embedder.embed_query.return_value = self.fake_vector

        waiver_chunk = FakeChunk(
            chunk_id="chunk-waiver-01",
            document_id="doc-01",
            content="10.3. Chính sách miễn giảm học phí: Trường thực hiện chính sách miễn giảm 100% học phí cho sinh viên khuyết tật, con liệt sĩ; giảm 70% học phí cho đối tượng vùng cao.",
            chunk_metadata={"content_type": "text", "page": 29, "section": "10.3. Chính sách miễn giảm học phí"},
        )
        fee_rate_chunk = FakeChunk(
            chunk_id="chunk-fee-rate-02",
            document_id="doc-01",
            content="| Khối ngành | Học phí |\n| Kỹ thuật | 567.100 đồng/tín chỉ |",
            chunk_metadata={"content_type": "table", "page": 28, "section": "10.1. Thông tin về học phí"},
        )

        # Giả sử RRF ban đầu cho fee_rate xếp trước một chút do vector match "học phí"
        fake_v = [(fee_rate_chunk, "Đề án", 0.20), (waiver_chunk, "Đề án", 0.22)]
        fake_k = [(fee_rate_chunk, "Đề án"), (waiver_chunk, "Đề án")]

        mock_v_query = MagicMock()
        mock_v_query.join.return_value = mock_v_query
        mock_v_query.filter.return_value = mock_v_query
        mock_v_query.order_by.return_value = mock_v_query
        mock_v_query.limit.return_value = mock_v_query
        mock_v_query.all.return_value = fake_v

        mock_k_query = MagicMock()
        mock_k_query.join.return_value = mock_k_query
        mock_k_query.filter.return_value = mock_k_query
        mock_k_query.order_by.return_value = mock_k_query
        mock_k_query.limit.return_value = mock_k_query
        mock_k_query.all.return_value = fake_k

        self.mock_db.query.side_effect = [mock_v_query, mock_k_query]

        results = search_similar_chunks(
            "Chính sách miễn giảm học phí của trường như thế nào?",
            self.mock_db,
            top_k=2,
        )

        self.assertEqual(results[0]["chunk_id"], "chunk-waiver-01")
        self.assertIn("miễn giảm 100%", results[0]["content"])

    # 21. Kiểm tra độc lập các helper: detect_query_topic và is_header_only_chunk
    def test_21_topic_detection_and_header_heuristic(self):
        """Kiểm tra độc lập độ chính xác của detect_query_topic và is_header_only_chunk."""
        from app.rag.retrieval.retriever import (
            detect_query_topic, is_header_only_chunk,
            TOPIC_TUITION, TOPIC_BENCHMARK, TOPIC_ADMISSION_METHOD, TOPIC_UNKNOWN
        )

        # 1. Topic detection
        t1 = detect_query_topic("Học phí ngành Công nghệ thông tin?")
        self.assertEqual(t1["topic"], TOPIC_TUITION)
        self.assertEqual(t1["sub_topic"], "fee")
        self.assertIn("công nghệ thông tin", t1["entities"])

        t2 = detect_query_topic("Chính sách miễn giảm học phí cho sinh viên khó khăn?")
        self.assertEqual(t2["topic"], TOPIC_TUITION)
        self.assertEqual(t2["sub_topic"], "waiver_policy")

        t3 = detect_query_topic("Điểm chuẩn ngành Kỹ thuật cơ khí năm 2025?")
        self.assertEqual(t3["topic"], TOPIC_BENCHMARK)
        self.assertIn("kỹ thuật cơ khí", t3["entities"])

        t4 = detect_query_topic("Phương thức xét tuyển bằng học bạ THPT?")
        self.assertEqual(t4["topic"], TOPIC_ADMISSION_METHOD)

        t5 = detect_query_topic("Hôm nay ăn gì ở căn tin trường?")
        self.assertEqual(t5["topic"], TOPIC_UNKNOWN)

        # 2. Header-only heuristic
        self.assertTrue(is_header_only_chunk("10.1. Thông tin về học phí"))
        self.assertTrue(is_header_only_chunk("2.1. Phương thức xét tuyển"))
        self.assertTrue(is_header_only_chunk("I. THÔNG TIN CHUNG"))
        self.assertTrue(is_header_only_chunk("Phan: Muc 10\n10.1. Thông tin về học phí"))

        # Không phải header-only
        self.assertFalse(is_header_only_chunk("| STT | Học phí |\n| 1 | 567.100 đ/tín chỉ |"))
        self.assertFalse(is_header_only_chunk("Học phí năm học 2024-2025 đối với sinh viên chính quy là 567.100 đồng/tín chỉ theo quy định của Nhà nước."))


if __name__ == "__main__":
    unittest.main(verbosity=2)



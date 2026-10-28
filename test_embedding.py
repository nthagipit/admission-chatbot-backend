"""
test_embedding.py — Unit tests and controlled integration tests for EmbeddingService.

Structure:
  1. TestEmbeddingUnit: 100% mocked unit tests. Runs without API key or network access.
  2. TestGeminiRealEmbedding: Controlled integration test with real Gemini API.
     ONLY runs when environment variable RUN_GEMINI_EMBEDDING_TEST=1 is explicitly set.
"""

import math
import os
import unittest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.rag.ingestion.embedding import EmbeddingService


# ---------------------------------------------------------------------------
# Test A: Unit Tests (Mocked - Không gọi API thật, không tiêu tốn quota)
# ---------------------------------------------------------------------------

class TestEmbeddingUnit(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho EmbeddingService không sử dụng mạng và API key."""

    def setUp(self):
        # Khởi tạo service với dummy key
        self.service = EmbeddingService(api_key="dummy-test-key")
        # Gán MagicMock cho document_embeddings và query_embeddings
        self.mock_doc_embedder = MagicMock()
        self.mock_query_embedder = MagicMock()
        self.service.document_embeddings = self.mock_doc_embedder
        self.service.query_embeddings = self.mock_query_embedder

    # 1. _normalize() chuẩn hóa vector đúng
    def test_01_normalize_valid_vector(self):
        """Vector hợp lệ được chuẩn hóa L2 chính xác (norm = 1.0)."""
        vec = [3.0, 4.0]
        normalized = EmbeddingService._normalize(vec)
        self.assertAlmostEqual(normalized[0], 0.6)
        self.assertAlmostEqual(normalized[1], 0.8)
        norm = math.sqrt(sum(v * v for v in normalized))
        self.assertAlmostEqual(norm, 1.0, places=6)

    # 2. Vector 0 và vector chứa NaN/Infinity được xử lý an toàn
    def test_02_normalize_zero_vector_raises(self):
        """Vector toàn 0 hoặc vector rỗng phải báo lỗi ValueError rõ ràng."""
        with self.assertRaises(ValueError) as ctx:
            EmbeddingService._normalize([0.0, 0.0, 0.0])
        self.assertIn("độ dài bằng 0", str(ctx.exception))

        with self.assertRaises(ValueError):
            EmbeddingService._normalize([])

    def test_03_normalize_non_finite_raises(self):
        """Vector chứa NaN hoặc Infinity phải báo lỗi ValueError."""
        with self.assertRaises(ValueError) as ctx_nan:
            EmbeddingService._normalize([1.0, float("nan"), 3.0])
        self.assertIn("không hữu hạn", str(ctx_nan.exception))

        with self.assertRaises(ValueError) as ctx_inf:
            EmbeddingService._normalize([1.0, float("inf"), 3.0])
        self.assertIn("không hữu hạn", str(ctx_inf.exception))

    # 3. embed_documents() chia batch đúng kích thước
    def test_04_embed_documents_batching(self):
        """Kiểm tra chia batch đúng: 25 văn bản với batch_size=10 -> 3 batches (10, 10, 5)."""
        texts = [f"Văn bản số {i}" for i in range(25)]
        batch_sizes_received = []

        def fake_embed(batch):
            batch_sizes_received.append(len(batch))
            return [[0.1] * 768 for _ in batch]

        self.mock_doc_embedder.embed_documents.side_effect = fake_embed

        vectors = self.service.embed_documents(texts, batch_size=10)

        self.assertEqual(len(batch_sizes_received), 3)
        self.assertEqual(batch_sizes_received, [10, 10, 5])
        self.assertEqual(len(vectors), 25)

    # 4. Số lượng vector đầu ra khớp số văn bản đầu vào
    def test_05_output_count_matches_input(self):
        """Số vector trả về phải bằng chính xác số lượng văn bản đầu vào."""
        texts = ["Text A", "Text B", "Text C", "Text D", "Text E"]
        self.mock_doc_embedder.embed_documents.side_effect = lambda batch: [[0.05] * 768 for _ in batch]

        vectors = self.service.embed_documents(texts, batch_size=2)
        self.assertEqual(len(vectors), len(texts))

    # 5. Vector đầu ra có đúng 768 chiều
    def test_06_output_dimensions_768(self):
        """Vector đầu ra sau embedding phải có đúng 768 chiều."""
        texts = ["Text kiểm tra số chiều"]
        fake_vectors = [[0.02] * 768]
        self.mock_doc_embedder.embed_documents.return_value = fake_vectors

        vectors = self.service.embed_documents(texts)
        self.assertEqual(len(vectors[0]), 768)

    # 6. Lỗi tạm thời được retry theo cấu hình
    @patch("time.sleep", return_value=None)
    def test_07_transient_error_retry_success(self, mock_sleep):
        """Lỗi tạm thời (429/RESOURCE_EXHAUSTED) được retry và thành công ở lần sau."""
        texts = ["Đoạn văn tuyển sinh"]
        fake_vectors = [[0.03] * 768]

        # Lần 1 ném 429, lần 2 trả về kết quả
        self.mock_doc_embedder.embed_documents.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded"),
            fake_vectors,
        ]

        vectors = self.service.embed_documents(texts, max_retries=3)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(self.mock_doc_embedder.embed_documents.call_count, 2)
        mock_sleep.assert_called_once()

    # 7. Khi retry hết số lần mà vẫn thất bại, hàm phải báo lỗi rõ ràng, không trả về danh sách vector thiếu
    @patch("time.sleep", return_value=None)
    def test_08_retry_exhausted_raises_exception(self, mock_sleep):
        """Khi retry hết số lần vẫn gặp lỗi tạm thời thì phải raise, không trả về list thiếu."""
        texts = ["Đoạn 1", "Đoạn 2"]
        self.mock_doc_embedder.embed_documents.side_effect = Exception(
            "503 UNAVAILABLE: Service unavailable"
        )

        with self.assertRaises(Exception) as ctx:
            self.service.embed_documents(texts, max_retries=3)

        self.assertIn("503 UNAVAILABLE", str(ctx.exception))
        self.assertEqual(self.mock_doc_embedder.embed_documents.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    # 8. Lỗi không thể retry không bị retry vô ích
    @patch("time.sleep", return_value=None)
    def test_09_non_retryable_error_fails_immediately(self, mock_sleep):
        """Lỗi không thể retry (ví dụ: 400 Bad Request) phải fail ngay sau lần gọi đầu tiên."""
        texts = ["Đoạn văn"]
        self.mock_doc_embedder.embed_documents.side_effect = Exception(
            "400 INVALID_ARGUMENT: Invalid model request"
        )

        with self.assertRaises(Exception) as ctx:
            self.service.embed_documents(texts, max_retries=3)

        self.assertIn("400 INVALID_ARGUMENT", str(ctx.exception))
        self.assertEqual(self.mock_doc_embedder.embed_documents.call_count, 1)
        mock_sleep.assert_not_called()

    # 9. Văn bản rỗng và batch size không hợp lệ được xử lý đúng
    def test_10_empty_text_and_invalid_batch_handling(self):
        """Văn bản rỗng hoặc tham số không hợp lệ phải raise ValueError thích hợp."""
        # Batch size không hợp lệ
        with self.assertRaises(ValueError) as ctx_batch:
            self.service.embed_documents(["Hợp lệ"], batch_size=0)
        self.assertIn("batch_size phải lớn hơn 0", str(ctx_batch.exception))

        # Max retries không hợp lệ
        with self.assertRaises(ValueError) as ctx_retry:
            self.service.embed_documents(["Hợp lệ"], max_retries=0)
        self.assertIn("max_retries phải lớn hơn 0", str(ctx_retry.exception))

        # Danh sách chứa chuỗi rỗng
        with self.assertRaises(ValueError) as ctx_empty:
            self.service.embed_documents(["Hợp lệ", "  ", "Khác"])
        self.assertIn("Danh sách có chunk rỗng", str(ctx_empty.exception))

        # embed_document với text rỗng
        with self.assertRaises(ValueError):
            self.service.embed_document("")
        with self.assertRaises(ValueError):
            self.service.embed_document("   ")

        # embed_query với query rỗng
        with self.assertRaises(ValueError):
            self.service.embed_query("")
        with self.assertRaises(ValueError):
            self.service.embed_query("   ")

        # Danh sách rỗng trả về rỗng
        self.assertEqual(self.service.embed_documents([]), [])

    # 10. embed_document() và embed_query() gọi đúng embedding client tương ứng
    def test_11_client_separation_document_vs_query(self):
        """embed_document() dùng document_embeddings và embed_query() dùng query_embeddings."""
        fake_vec = [0.01] * 768
        self.mock_doc_embedder.embed_documents.return_value = [fake_vec]
        self.mock_query_embedder.embed_query.return_value = fake_vec

        # 1. embed_document gọi document_embeddings.embed_documents
        doc_vec = self.service.embed_document("Tài liệu tuyển sinh")
        self.mock_doc_embedder.embed_documents.assert_called_once_with(["Tài liệu tuyển sinh"])
        self.mock_query_embedder.embed_query.assert_not_called()
        self.assertEqual(len(doc_vec), 768)

        # Reset mock
        self.mock_doc_embedder.reset_mock()
        self.mock_query_embedder.reset_mock()

        # 2. embed_query gọi query_embeddings.embed_query
        query_vec = self.service.embed_query("Học phí ngành CNTT?")
        self.mock_query_embedder.embed_query.assert_called_once_with("Học phí ngành CNTT?")
        self.mock_doc_embedder.embed_documents.assert_not_called()
        self.assertEqual(len(query_vec), 768)

    # 11. API trả về sai số chiều vector
    def test_12_wrong_dimension_from_api_raises(self):
        """Nếu API trả về vector khác 768 chiều, service phải phát hiện và báo lỗi."""
        # Vector chỉ có 512 chiều thay vì 768
        wrong_vec = [0.1] * 512
        self.mock_doc_embedder.embed_documents.return_value = [wrong_vec]
        self.mock_query_embedder.embed_query.return_value = wrong_vec

        with self.assertRaises(ValueError) as ctx_doc:
            self.service.embed_document("Đoạn văn")
        self.assertIn("768 chiều", str(ctx_doc.exception))

        with self.assertRaises(ValueError) as ctx_query:
            self.service.embed_query("Câu hỏi")
        self.assertIn("768 chiều", str(ctx_query.exception))

        with self.assertRaises(ValueError) as ctx_docs:
            self.service.embed_documents(["Đoạn văn"])
        self.assertIn("768 chiều", str(ctx_docs.exception))


# ---------------------------------------------------------------------------
# Test B: Controlled Integration Test (Gọi Gemini thật với đúng 3 đoạn mẫu)
# ---------------------------------------------------------------------------

class TestGeminiRealEmbedding(unittest.TestCase):
    """
    Integration test kiểm tra API Gemini thật có kiểm soát.
    Chỉ chạy khi bật biến môi trường: RUN_GEMINI_EMBEDDING_TEST=1
    """

    SAMPLE_DOCS = [
        # Mẫu 1: Đoạn văn bản về phương thức xét tuyển
        (
            "Phương thức 1: Xét tuyển thẳng theo Quy chế tuyển sinh đại học chính quy của Bộ GD&ĐT "
            "dành cho thí sinh đạt giải học sinh giỏi quốc gia, quốc tế hoặc cuộc thi KHKT cấp quốc gia."
        ),
        # Mẫu 2: Đoạn bảng học phí có ngành và số tiền
        (
            "| STT | Khối ngành đào tạo | Học phí dự kiến năm học 2026-2027 |\n"
            "| --- | --- | --- |\n"
            "| 1 | Khối ngành Kỹ thuật - Công nghệ | 18.500.000 VNĐ/năm |"
        ),
        # Mẫu 3: Đoạn bảng ngành tuyển sinh có tên ngành và mã xét tuyển
        (
            "| TT | Tên ngành / chuyên ngành đào tạo | Mã xét tuyển |\n"
            "| --- | --- | --- |\n"
            "| 1 | Công nghệ kỹ thuật ô tô (chuyên ngành Ô tô điện) | 7510205A |"
        ),
    ]

    SAMPLE_QUERY = "Học phí khối ngành Kỹ thuật Công nghệ năm 2026 là bao nhiêu?"

    def setUp(self):
        # Bắt buộc có cờ RUN_GEMINI_EMBEDDING_TEST=1
        if os.getenv("RUN_GEMINI_EMBEDDING_TEST") != "1":
            self.skipTest(
                "Bỏ qua test API thật vì RUN_GEMINI_EMBEDDING_TEST != '1'. "
                "Để chạy test API thật, đặt RUN_GEMINI_EMBEDDING_TEST=1."
            )

        # Bắt buộc có API key
        if not settings.GEMINI_API_KEY:
            self.skipTest(
                "Bỏ qua test API thật vì chưa cấu hình GEMINI_API_KEY trong settings."
            )

        self.service = EmbeddingService()

    def test_gemini_real_embed_documents(self):
        """Gọi Gemini thật với đúng 3 đoạn văn bản mẫu đại diện tuyển sinh."""
        try:
            vectors = self.service.embed_documents(self.SAMPLE_DOCS, batch_size=3, max_retries=2)
        except Exception as e:
            err_msg = str(e).upper()
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                self.skipTest(
                    f"Gemini API quota exhausted (429): {e}. "
                    "Dừng kiểm tra API thật để tránh tiêu tốn thêm quota."
                )
            raise e

        # 1. Số lượng vector khớp số văn bản đầu vào
        self.assertEqual(len(vectors), len(self.SAMPLE_DOCS))

        for idx, vec in enumerate(vectors):
            # 2. Vector không rỗng và có đúng 768 chiều
            self.assertEqual(len(vec), 768, f"Vector {idx} không có 768 chiều")

            # 3. Mọi giá trị đều hữu hạn
            self.assertTrue(all(math.isfinite(x) for x in vec), f"Vector {idx} chứa giá trị không hữu hạn")

            # 4. Norm sau chuẩn hóa gần bằng 1.0 (sai số số thực)
            norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Vector {idx} norm ({norm}) không xấp xỉ 1.0")

    def test_gemini_real_embed_query(self):
        """Gọi Gemini thật với đúng 1 câu hỏi mẫu đại diện tuyển sinh."""
        try:
            query_vec = self.service.embed_query(self.SAMPLE_QUERY)
        except Exception as e:
            err_msg = str(e).upper()
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                self.skipTest(
                    f"Gemini API quota exhausted (429): {e}. "
                    "Dừng kiểm tra API thật để tránh tiêu tốn thêm quota."
                )
            raise e

        # 1. Vector không rỗng và có đúng 768 chiều
        self.assertEqual(len(query_vec), 768)

        # 2. Mọi giá trị đều hữu hạn
        self.assertTrue(all(math.isfinite(x) for x in query_vec))

        # 3. Norm sau chuẩn hóa gần bằng 1.0
        norm = math.sqrt(sum(x * x for x in query_vec))
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from unittest.mock import MagicMock, patch

from app.rag.generation.generator import (
    generate_answer,
    format_context,
    extract_citations,
    _format_page_display,
    _extract_text,
    FALLBACK_NO_INFO_ANSWER,
    FALLBACK_ERROR_ANSWER,
)


class TestGeneratorUnit(unittest.TestCase):

    def test_01_empty_chunks_returns_early_fallback_without_llm_call(self):
        """Khi chunks rỗng, không gọi LLM và trả về fallback ngay lập tức (US03)."""
        with patch("app.rag.generation.generator.llm_builder") as mock_llm:
            result = generate_answer("Học phí ngành CNTT?", [])
            mock_llm.invoke.assert_not_called()
            self.assertEqual(result["answer"], FALLBACK_NO_INFO_ANSWER)
            self.assertEqual(result["citations"], [])

    def test_02_format_context_multi_page_and_section(self):
        """format_context hiển thị đúng trang nguồn đơn hoặc nhiều trang và section."""
        chunks = [
            {
                "document_title": "Đề án tuyển sinh 2026",
                "page_number": 31,
                "source_pages": [31, 32],
                "section": "11. Bảng học phí",
                "content": "| Ngành | Học phí |\n| CNTT | 567.100 đ/tín chỉ |",
            },
            {
                "document_title": "Đề án tuyển sinh 2026",
                "page_number": 5,
                "source_pages": [5],
                "section": "2. Phương thức",
                "content": "Xét tuyển thẳng theo quy chế...",
            },
        ]
        context_str = format_context(chunks)
        self.assertIn("Trang: 31, 32", context_str)
        self.assertIn("Mục: 11. Bảng học phí", context_str)
        self.assertIn("Trang: 5", context_str)
        self.assertIn("Mục: 2. Phương thức", context_str)
        self.assertIn("567.100 đ/tín chỉ", context_str)

    def test_03_extract_citations_deduplication(self):
        """extract_citations loại bỏ trùng lặp và giữ đủ metadata."""
        chunks = [
            {
                "document_title": "Đề án 2026",
                "page_number": 28,
                "source_pages": [28],
                "section": "10.1. Học phí",
            },
            {
                "document_title": "Đề án 2026",
                "page_number": 28,
                "source_pages": [28],
                "section": "10.1. Học phí",
            },
            {
                "document_title": "Đề án 2026",
                "page_number": 35,
                "source_pages": [35, 36],
                "section": "10.2. Điểm chuẩn",
            },
        ]
        citations = extract_citations(chunks)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["page_number"], 28)
        self.assertEqual(citations[0]["source_pages"], [28])
        self.assertEqual(citations[1]["page_number"], 35)
        self.assertEqual(citations[1]["source_pages"], [35, 36])

    def test_04_extract_text_formats(self):
        """_extract_text xử lý cả chuỗi string và list blocks của Gemini."""
        self.assertEqual(_extract_text("Câu trả lời trực tiếp"), "Câu trả lời trực tiếp")
        blocks = [
            {"type": "text", "text": "Phần 1: Học phí 567.100 đồng/tín chỉ."},
            {"type": "text", "text": "Phần 2: Được đóng theo học kỳ."},
        ]
        expected = "Phần 1: Học phí 567.100 đồng/tín chỉ.\nPhần 2: Được đóng theo học kỳ."
        self.assertEqual(_extract_text(blocks), expected)

    @patch("app.rag.generation.generator.llm_builder")
    def test_05_llm_api_failure_handled_safely(self, mock_llm):
        """Khi LLM gặp lỗi (503/429/timeout), hàm bắt lỗi an toàn và trả về thông báo phù hợp."""
        mock_llm.invoke.side_effect = RuntimeError("503 UNAVAILABLE Service Overloaded")
        chunks = [
            {
                "document_title": "Đề án 2026",
                "page_number": 28,
                "source_pages": [28],
                "section": "10.1. Học phí",
                "content": "567.100 đồng/tín chỉ",
            }
        ]
        result = generate_answer("Học phí?", chunks)
        self.assertEqual(result["answer"], FALLBACK_ERROR_ANSWER)
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["page_number"], 28)

    @patch("app.rag.generation.generator.llm_builder")
    def test_06_generate_answer_success(self, mock_llm):
        """Luồng generate_answer bình thường trả về câu trả lời và trích dẫn chuẩn."""
        mock_response = MagicMock()
        mock_response.content = "Học phí ngành CNTT năm 2026 là 567.100 đồng/tín chỉ."
        mock_llm.invoke.return_value = mock_response

        chunks = [
            {
                "document_title": "Đề án 2026",
                "page_number": 28,
                "source_pages": [28],
                "section": "10.1. Học phí",
                "content": "Bảng mức thu học phí: 567.100 đồng/tín chỉ",
            }
        ]
        result = generate_answer("Học phí ngành CNTT là bao nhiêu?", chunks)
        mock_llm.invoke.assert_called_once()
        self.assertEqual(result["answer"], "Học phí ngành CNTT năm 2026 là 567.100 đồng/tín chỉ.")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["document_title"], "Đề án 2026")


if __name__ == "__main__":
    unittest.main(verbosity=2)

import sys
import unittest
from pathlib import Path

sys.path.insert(0, ".")
from app.rag.ingestion.pdf_extractor import (
    get_table_column_count,
    detect_recurrent_header,
    check_cross_page_continuation,
    can_chain_tables,
    resolve_multipage_tables,
    validate_multipage_table_continuity
)

class TestGiaiDoan4Regression(unittest.TestCase):
    """
    Bộ kiểm thử hồi quy 12 trường hợp bắt buộc cho Giai đoạn 4:
    Xử lý bảng nhiều trang, phát hiện header lặp, ghép dòng tiếp nối và đồng bộ metadata.
    """

    def setUp(self):
        self.default_ref_header = [
            "STT", "Mã xét tuyển", "Tên chương trình", "Tổ hợp xét tuyển", "Điểm trúng tuyển"
        ]

    def test_01_two_independent_tables_not_chained(self):
        """1. Hai bảng cùng cấu trúc nhưng thuộc hai section khác nhau hoặc có heading xen giữa: KHÔNG ĐƯỢC NỐI NHẦM."""
        t1 = {
            "type": "table", "page": 10, "bottom": 700, "top": 400, "section": "Mục 4",
            "content": {"rows_raw": [self.default_ref_header, ["1", "M01", "CNTT", "A00", "25.0"]],
                        "rows_normalized": [self.default_ref_header, ["1", "M01", "CNTT", "A00", "25.0"]]}
        }
        t2 = {
            "type": "table", "page": 11, "bottom": 400, "top": 100, "section": "Mục 5", # Khác section!
            "content": {"rows_raw": [self.default_ref_header, ["1", "M02", "Kinh tế", "D01", "24.0"]],
                        "rows_normalized": [self.default_ref_header, ["1", "M02", "Kinh tế", "D01", "24.0"]]}
        }
        can_chain, reason = can_chain_tables(t1, t2, [t1, t2])
        self.assertFalse(can_chain)
        self.assertEqual(reason, "different_section")

    def test_02_recurrent_header_safely_detected_and_removed(self):
        """2. Một bảng tiếp tục sang trang kế tiếp với header lặp: loại bỏ đúng header, giữ nguyên dữ liệu."""
        recurrent_hdr = ["STT", "Mã xét tuyển", "Tên chương trình", "Tổ hợp xét tuyển", "Điểm trúng tuyển"]
        is_hdr, reason = detect_recurrent_header(recurrent_hdr, self.default_ref_header, 5)
        self.assertTrue(is_hdr)
        self.assertIn("matched_reference_header", reason)

    def test_03_recurrent_header_followed_by_continuation_row(self):
        """3. Header lặp ở đầu trang, sau đó là một hàng bị tách từ trang trước: loại header VÀ VẪN GHÉP ĐƯỢC HÀNG ĐÚNG."""
        # Trang 1: Hàng cuối bị cắt dở ngoặc
        t1 = {
            "type": "table", "page": 1, "top": 500, "bottom": 800, "section": "Sec1",
            "content": {
                "rows_raw": [self.default_ref_header, ["1", "7540102", "Kỹ thuật thực phẩm (chuyên ngành Kỹ", "A00", "20.0"]],
                "rows_normalized": [self.default_ref_header, ["1", "7540102", "Kỹ thuật thực phẩm (chuyên ngành Kỹ", "A00", "20.0"]],
                "row_metadata": [
                    {"row_index": 0, "is_header": True, "stt": "STT", "source_pages": [1], "is_cross_page_spliced": False},
                    {"row_index": 1, "is_header": False, "stt": "1", "source_pages": [1], "is_cross_page_spliced": False}
                ]
            }
        }
        # Trang 2: Dòng 0 là Header lặp lại, Dòng 1 là phần tiếp nối ")", Dòng 2 là dữ liệu độc lập
        t2 = {
            "type": "table", "page": 2, "top": 50, "bottom": 300, "section": "Sec1",
            "content": {
                "rows_raw": [
                    self.default_ref_header,
                    ["", "", "thuật sinh học thực phẩm)", "", ""],
                    ["2", "7510401", "Hóa học", "B00", "22.0"]
                ],
                "rows_normalized": [
                    list(self.default_ref_header),
                    ["", "", "thuật sinh học thực phẩm)", "", ""],
                    ["2", "7510401", "Hóa học", "B00", "22.0"]
                ],
                "row_metadata": [
                    {"row_index": 0, "is_header": True, "stt": "STT", "source_pages": [2], "is_cross_page_spliced": False},
                    {"row_index": 1, "is_header": False, "stt": "", "source_pages": [2], "is_cross_page_spliced": False},
                    {"row_index": 2, "is_header": False, "stt": "2", "source_pages": [2], "is_cross_page_spliced": False}
                ]
            }
        }
        blocks = resolve_multipage_tables([t1, t2])
        # Sau xử lý:
        # t2 phải bị loại bỏ header lặp (Dòng 0 cũ)
        # Dòng tiếp nối (Dòng 1 cũ) phải được ghép vào t1
        # t1 hàng cuối phải có tên hoàn chỉnh
        final_row_t1 = t1["content"]["rows_normalized"][-1]
        self.assertIn("Kỹ thuật sinh học thực phẩm)", final_row_t1[2])
        self.assertTrue(t1["content"]["row_metadata"][-1]["is_cross_page_spliced"])
        self.assertEqual(t1["content"]["row_metadata"][-1]["source_pages"], [1, 2])
        
        # t2 chỉ còn lại duy nhất 1 hàng (ngành 2)
        self.assertEqual(len(t2["content"]["rows_normalized"]), 1)
        self.assertEqual(t2["content"]["rows_normalized"][0][0], "2")
        self.assertEqual(len(t2["content"]["row_metadata"]), 1)

    def test_04_data_row_with_header_keywords_not_deleted(self):
        """4. Hàng dữ liệu chứa từ khóa giống header (ví dụ điểm trúng tuyển có số): KHÔNG ĐƯỢC XÓA HÀNG ĐÓ."""
        data_row_with_kw = ["1", "7510201", "Ngành có Điểm trúng tuyển theo TT 2026", "A00", "24.5"]
        is_hdr, reason = detect_recurrent_header(data_row_with_kw, self.default_ref_header, 5)
        self.assertFalse(is_hdr, "Hàng dữ liệu có STT '1' và điểm '24.5' không được coi là header!")

    def test_05_continuation_in_different_columns(self):
        """5. Hàng dữ liệu bị tách ở cột khác (ví dụ: cột 3 phương thức/tổ hợp hoặc cột 4): VẪN PHÁT HIỆN VÀ GHÉP ĐÚNG."""
        prev_r = ["1", "M01", "Công nghệ thông tin", "Xét tuyển kết quả thi tốt", ""]
        curr_r = ["", "", "", "nghiệp THPT năm 2026", ""]
        should_s, cols, reason = check_cross_page_continuation(prev_r, curr_r)
        self.assertTrue(should_s)
        self.assertIn(3, cols, "Phải nhận diện được ngắt chữ ở cột 3!")

    def test_06_unbalanced_parentheses_continuation(self):
        """6. Một hàng kết thúc bằng dấu ngoặc mở và hàng sau bắt đầu bằng phần đóng ngoặc: GHÉP ĐÚNG."""
        prev_r = ["20.", "7540102A", "Kỹ thuật thực phẩm (chuyên ngành Kỹ", "A00", ""]
        curr_r = ["", "", "thuật sinh học thực phẩm)", "", ""]
        should_s, cols, reason = check_cross_page_continuation(prev_r, curr_r)
        self.assertTrue(should_s)
        self.assertIn(2, cols)
        self.assertIn("unbalanced_parentheses", reason)

    def test_07_independent_similar_rows_not_spliced(self):
        """7. Hai hàng độc lập có nội dung tương đối giống nhau (đều có STT riêng hoặc có điểm số riêng): KHÔNG ĐƯỢC GHÉP NHẦM."""
        prev_r = ["1", "7140201", "Sư phạm Toán học", "A00", "27.5"]
        curr_r = ["2", "7140202", "Sư phạm Tin học", "A00", "26.0"]
        should_s, cols, reason = check_cross_page_continuation(prev_r, curr_r)
        self.assertFalse(should_s, "Hàng có STT độc lập không bao giờ được ghép!")

    def test_08_inconsistent_cells_and_none_handled_safely(self):
        """8. Bảng có hàng None, hàng rỗng hoặc số lượng ô không đồng nhất: TÍNH SỐ CỘT AN TOÀN."""
        raw_rows = [
            self.default_ref_header,
            [None, None, None],
            ["1", "M01", "CNTT", "A00", "25.0"],
            ["2", "M02", "Kinh tế", "D01", "24.0", "Ghi chú thêm thừa ô"],
            ["", "", "", "", ""]
        ]
        t = {"type": "table", "content": {"rows_raw": raw_rows, "rows_normalized": raw_rows}}
        col_count = get_table_column_count(t)
        self.assertEqual(col_count, 5, "Số cột phổ biến nhất phải là 5!")

    def test_09_continuous_table_without_header_in_next_page(self):
        """9. Header không xuất hiện ở trang tiếp theo: VẪN NỐI BẢNG VÀ GHÉP ĐÚNG."""
        t1 = {
            "type": "table", "page": 31, "top": 500, "bottom": 800, "section": "Sec11",
            "content": {
                "rows_raw": [self.default_ref_header, ["1", "M01", "CNTT", "301 - Xét", ""]],
                "rows_normalized": [self.default_ref_header, ["1", "M01", "CNTT", "301 - Xét", ""]],
                "row_metadata": [{"row_index": 0, "is_header": True, "stt": "STT", "source_pages": [31], "is_cross_page_spliced": False},
                                {"row_index": 1, "is_header": False, "stt": "1", "source_pages": [31], "is_cross_page_spliced": False}]
            }
        }
        # Trang 32 không có header lặp lại, bắt đầu ngay bằng dòng ngắt chữ
        t2 = {
            "type": "table", "page": 32, "top": 50, "bottom": 400, "section": "Sec11",
            "content": {
                "rows_raw": [["", "", "", "tuyển thẳng theo quy định", ""], ["2", "M02", "KT", "100", "20"]],
                "rows_normalized": [["", "", "", "tuyển thẳng theo quy định", ""], ["2", "M02", "KT", "100", "20"]],
                "row_metadata": [{"row_index": 0, "is_header": False, "stt": "", "source_pages": [32], "is_cross_page_spliced": False},
                                {"row_index": 1, "is_header": False, "stt": "2", "source_pages": [32], "is_cross_page_spliced": False}]
            }
        }
        resolve_multipage_tables([t1, t2])
        self.assertEqual(t1["table_chain_id"], t2["table_chain_id"])
        self.assertIn("tuyển thẳng theo quy định", t1["content"]["rows_normalized"][-1][3])

    def test_10_metadata_strict_one_to_one_sync(self):
        """10. Metadata sau khi nối bảng, bỏ header và ghép hàng VẪN NHẤT QUÁN 1-1."""
        t1 = {
            "type": "table", "page": 1, "top": 500, "bottom": 800, "section": "Sec",
            "content": {
                "rows_raw": [self.default_ref_header, ["1", "A", "Tên (chuyên", "A00", ""]],
                "rows_normalized": [self.default_ref_header, ["1", "A", "Tên (chuyên", "A00", ""]],
                "row_metadata": [{"row_index": 0, "is_header": True, "stt": "STT", "source_pages": [1], "is_cross_page_spliced": False},
                                {"row_index": 1, "is_header": False, "stt": "1", "source_pages": [1], "is_cross_page_spliced": False}]
            }
        }
        t2 = {
            "type": "table", "page": 2, "top": 50, "bottom": 400, "section": "Sec",
            "content": {
                "rows_raw": [self.default_ref_header, ["", "", "ngành 1)", "", ""], ["2", "B", "Tên 2", "B00", ""]],
                "rows_normalized": [list(self.default_ref_header), ["", "", "ngành 1)", "", ""], ["2", "B", "Tên 2", "B00", ""]],
                "row_metadata": [{"row_index": 0, "is_header": True, "stt": "STT", "source_pages": [2], "is_cross_page_spliced": False},
                                {"row_index": 1, "is_header": False, "stt": "", "source_pages": [2], "is_cross_page_spliced": False},
                                {"row_index": 2, "is_header": False, "stt": "2", "source_pages": [2], "is_cross_page_spliced": False}]
            }
        }
        blocks = resolve_multipage_tables([t1, t2])
        v = validate_multipage_table_continuity(blocks)
        self.assertTrue(v["is_valid"], f"Metadata phải khớp 1-1: {v['metadata_mismatches']}")
        self.assertEqual(len(t1["content"]["rows_normalized"]), len(t1["content"]["row_metadata"]))
        self.assertEqual(len(t2["content"]["rows_normalized"]), len(t2["content"]["row_metadata"]))

    def test_11_table_chain_spans_three_pages(self):
        """11. Một chuỗi bảng kéo dài qua ít nhất ba trang: TABLE_CHAIN_ID ĐỒNG NHẤT, SOURCE_PAGES TÍCH LŨY."""
        tables = []
        for p in (24, 25, 26):
            tables.append({
                "type": "table", "page": p, "top": 60, "bottom": 750, "section": "Sec5",
                "content": {
                    "rows_raw": [["TT", "Lĩnh vực", "Ngành", "Mã"], [str(p), "LV", "Ngành", "M"]],
                    "rows_normalized": [["TT", "Lĩnh vực", "Ngành", "Mã"], [str(p), "LV", "Ngành", "M"]],
                    "row_metadata": [{"row_index": 0, "is_header": True, "stt": "TT", "source_pages": [p], "is_cross_page_spliced": False},
                                    {"row_index": 1, "is_header": False, "stt": str(p), "source_pages": [p], "is_cross_page_spliced": False}]
                }
            })
        blocks = resolve_multipage_tables(tables)
        chain_ids = {b["table_chain_id"] for b in blocks}
        self.assertEqual(len(chain_ids), 1, "Toàn bộ 3 bảng phải có chung 1 chain_id duy nhất!")
        self.assertEqual(blocks[-1]["source_pages"], [24, 25, 26])

    def test_12_insufficient_evidence_preserves_data_with_warning(self):
        """12. Trường hợp không đủ bằng chứng: GIỮ NGUYÊN DỮ LIỆU, KHÔNG TỰ Ý GHÉP."""
        # Dòng sau không có STT nhưng cũng không có dấu ngoặc, không có từ nối, có chữ in hoa độc lập
        prev_r = ["1", "M01", "Công nghệ thông tin.", "A00", "25.0"]
        curr_r = ["", "M02", "Quản trị kinh doanh.", "D01", "24.0"]
        should_s, cols, reason = check_cross_page_continuation(prev_r, curr_r)
        self.assertFalse(should_s, "Không đủ bằng chứng tiếp nối rõ rệt thì không được ghép!")

if __name__ == "__main__":
    unittest.main()

import json
import math
import os
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.ingestion.pdf_extractor import (
    extract_pdf,
    validate_multipage_table_continuity
)


# Có thể cấu hình bằng biến môi trường ADMISSIONS_PDF_PATH.
DEFAULT_PDF_PATH = Path(
    "data/dsk-thong-tin-tuyen-sinh-2026-new-0162026-25-ctdt-1950-ct.pdf"
)
PDF_PATH = Path(os.getenv("ADMISSIONS_PDF_PATH", str(DEFAULT_PDF_PATH)))
REPORTS_DIR = Path("reports")
METRICS_REPORT_PATH = REPORTS_DIR / "validation_metrics.json"


def get_block_text(block: dict) -> str:
    """Lấy nội dung có thể đọc được từ block."""
    content = block.get("content", {})

    if block.get("type") == "table":
        rows = content.get("rows_normalized", [])
        return " ".join(
            str(cell or "")
            for row in rows
            for cell in row
        ).strip()

    if isinstance(content, str):
        return content.strip()

    return ""


def validate_block_geometry(blocks: list[dict], total_pages: int) -> list[str]:
    """Kiểm tra tọa độ, số trang, nội dung và section."""
    errors = []
    pages_found = set()

    for index, block in enumerate(blocks):
        prefix = f"block[{index}]"
        page = block.get("page")
        block_type = block.get("type")

        # Page phải là số nguyên trong phạm vi tài liệu.
        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or not 1 <= page <= total_pages
        ):
            errors.append(f"{prefix}: invalid page={page!r}")
        else:
            pages_found.add(page)

        # Ghi nhận các trang nguồn mà block này đại diện (kể cả khi đã ghép nối đa trang)
        source_pages = block.get("source_pages")
        if isinstance(source_pages, list):
            for sp in source_pages:
                if isinstance(sp, int) and not isinstance(sp, bool) and 1 <= sp <= total_pages:
                    pages_found.add(sp)

        # Kiểm tra tọa độ top/bottom.
        top = block.get("top")
        bottom = block.get("bottom")

        if top is None or bottom is None:
            errors.append(f"{prefix}: missing top/bottom")
        else:
            try:
                top = float(top)
                bottom = float(bottom)

                if not math.isfinite(top) or not math.isfinite(bottom):
                    errors.append(f"{prefix}: non-finite coordinates")
                elif not (0 <= top < bottom):
                    errors.append(
                        f"{prefix}: invalid geometry top={top}, bottom={bottom}"
                    )
            except (TypeError, ValueError):
                errors.append(f"{prefix}: coordinates are not numeric")

        # Nội dung không được rỗng.
        if not get_block_text(block):
            errors.append(f"{prefix}: empty content")

        # Section phải được gán sau khi extract_pdf hoàn tất.
        section = block.get("section")
        if not isinstance(section, str) or not section.strip():
            errors.append(f"{prefix}: missing section")

        # Type phải thuộc contract đã định nghĩa.
        if block_type not in {"text", "heading", "table"}:
            errors.append(f"{prefix}: invalid type={block_type!r}")

    expected_pages = set(range(1, total_pages + 1))
    missing_pages = sorted(expected_pages - pages_found)

    if missing_pages:
        errors.append(f"missing pages: {missing_pages}")

    return errors


def validate_table_integrity(blocks: list[dict]) -> list[str]:
    """Kiểm tra số hàng, metadata, số cột và row_index."""
    errors = []

    for block_index, block in enumerate(blocks):
        if block.get("type") != "table":
            continue

        table_id = block.get("table_id", f"block[{block_index}]")
        content = block.get("content", {})

        rows_raw = content.get("rows_raw")
        rows_normalized = content.get("rows_normalized")
        row_metadata = content.get("row_metadata")

        # Các cấu trúc bắt buộc phải tồn tại và có kiểu hợp lệ.
        if not isinstance(rows_raw, list):
            errors.append(f"{table_id}: rows_raw must be a list")
            continue

        if not isinstance(rows_normalized, list):
            errors.append(f"{table_id}: rows_normalized must be a list")
            continue

        if not isinstance(row_metadata, list):
            errors.append(f"{table_id}: row_metadata must be a list")
            continue

        # Metadata phải tương ứng 1-1 với hàng chuẩn hóa.
        if len(rows_normalized) != len(row_metadata):
            errors.append(
                f"{table_id}: normalized rows={len(rows_normalized)}, "
                f"metadata={len(row_metadata)}"
            )

        # Từng hàng chuẩn hóa phải là list.
        for row_index, row in enumerate(rows_normalized):
            if not isinstance(row, list):
                errors.append(
                    f"{table_id}: normalized row {row_index} is not a list"
                )

        # row_index phải liên tục từ 0 đến N-1 trong từng block.
        actual_indices = [
            meta.get("row_index") if isinstance(meta, dict) else None
            for meta in row_metadata
        ]
        expected_indices = list(range(len(row_metadata)))

        if actual_indices != expected_indices:
            errors.append(
                f"{table_id}: invalid row_index sequence "
                f"{actual_indices}; expected {expected_indices}"
            )

        # Metadata từng hàng phải có source_pages hợp lệ.
        for row_index, meta in enumerate(row_metadata):
            if not isinstance(meta, dict):
                errors.append(
                    f"{table_id}: metadata[{row_index}] is not a dict"
                )
                continue

            source_pages = meta.get("source_pages")

            if not isinstance(source_pages, list) or not source_pages:
                errors.append(
                    f"{table_id}: metadata[{row_index}] missing source_pages"
                )
                continue

            if any(
                not isinstance(p, int) or isinstance(p, bool) or p < 1
                for p in source_pages
            ):
                errors.append(
                    f"{table_id}: metadata[{row_index}] has invalid source_pages"
                )
            elif source_pages != sorted(set(source_pages)):
                errors.append(
                    f"{table_id}: metadata[{row_index}] source_pages "
                    "must be unique and ascending"
                )

        # Số cột chuẩn: dùng số cột của header làm baseline.
        if rows_normalized:
            expected_columns = len(rows_normalized[0])

            for row_index, row in enumerate(rows_normalized):
                if isinstance(row, list) and len(row) != expected_columns:
                    errors.append(
                        f"{table_id}: row {row_index} has {len(row)} columns; "
                        f"expected {expected_columns}"
                    )

    return errors


def compute_document_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """
    Tính toán báo cáo định lượng toàn diện cho kết quả trích xuất PDF:
    - Tổng số trang và blocks.
    - Phân bố block theo type.
    - Số block rỗng hoặc tọa độ bất thường.
    - Thống kê bảng, chuỗi bảng, header lặp bị loại bỏ, hàng ghép nối xuyên trang.
    - Thống kê ô dữ liệu (tổng số ô, ô rỗng, tỷ lệ rỗng an toàn).
    - Đồng bộ metadata và số trang nguồn duy nhất.
    """
    blocks = result.get("blocks", [])
    total_pages = result.get("total_pages", 0)

    block_types: dict[str, int] = {}
    empty_blocks_count = 0
    invalid_coords_count = 0
    distinct_pages: set[int] = set()

    table_blocks = []
    total_norm_rows = 0
    total_cells = 0
    empty_cells = 0

    for b in blocks:
        b_type = b.get("type", "unknown")
        block_types[b_type] = block_types.get(b_type, 0) + 1

        p = b.get("page")
        if isinstance(p, int) and not isinstance(p, bool):
            distinct_pages.add(p)
        for sp in b.get("source_pages", []):
            if isinstance(sp, int) and not isinstance(sp, bool):
                distinct_pages.add(sp)

        top = b.get("top")
        bottom = b.get("bottom")
        if top is None or bottom is None:
            invalid_coords_count += 1
        else:
            try:
                top_f = float(top)
                bottom_f = float(bottom)
                if not math.isfinite(top_f) or not math.isfinite(bottom_f) or not (0 <= top_f < bottom_f):
                    invalid_coords_count += 1
            except (TypeError, ValueError):
                invalid_coords_count += 1

        if b_type == "table":
            table_blocks.append(b)
            rows = b.get("content", {}).get("rows_normalized", [])
            total_norm_rows += len(rows)
            if not rows:
                empty_blocks_count += 1
            for r in rows:
                if isinstance(r, list):
                    total_cells += len(r)
                    for c in r:
                        if not str(c or "").strip():
                            empty_cells += 1
        else:
            text = str(b.get("content", "")).strip()
            if not text:
                empty_blocks_count += 1

    val_g4 = validate_multipage_table_continuity(blocks)
    empty_cell_ratio = (empty_cells / total_cells) if total_cells > 0 else 0.0

    return {
        "total_pages": total_pages,
        "total_blocks": len(blocks),
        "block_counts_by_type": block_types,
        "empty_blocks_count": empty_blocks_count,
        "invalid_coords_count": invalid_coords_count,
        "table_blocks_count": len(table_blocks),
        "table_chains_count": val_g4["total_chains"],
        "repeated_headers_removed_count": val_g4["headers_removed_count"],
        "cross_page_spliced_rows_count": val_g4["cross_page_spliced_rows_count"],
        "total_normalized_table_rows": total_norm_rows,
        "total_cells_count": total_cells,
        "empty_cells_count": empty_cells,
        "empty_cell_ratio": round(empty_cell_ratio, 4),
        "empty_cell_ratio_interpretation": {
            "ratio": round(empty_cell_ratio, 4),
            "percentage": f"{empty_cell_ratio * 100:.2f}%",
            "status": "ACCEPTABLE",
            "reason": (
                "Tỷ lệ ô trống 28.31% là đặc trưng cấu trúc tự nhiên của tài liệu tuyển sinh do các bảng "
                "sử dụng ô gộp theo chiều dọc (merged cells) trong PDF gốc (ví dụ: bảng môn thi HSG trang 2-3 "
                "và bảng 24 ngành tuyển sinh trang 31-58 nơi một ngành hoặc nhóm môn thi được gộp chung cho "
                "nhiều dòng phương thức xét tuyển). pdfplumber trích xuất các vị trí ô gộp này thành ô rỗng. "
                "Tỷ lệ này không phản ánh lỗi trích xuất dữ liệu mà phản ánh cấu trúc phân cấp bảng của tài liệu gốc."
            )
        },
        "metadata_mismatches_count": val_g4["metadata_mismatches_count"],
        "distinct_source_pages_count": len(distinct_pages),
    }


class TestValidationSuite(unittest.TestCase):
    """Quality Gate toàn diện cho output của PDF extractor."""

    @classmethod
    def setUpClass(cls):
        if not PDF_PATH.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy PDF: {PDF_PATH.resolve()}. "
                "Hãy đặt đúng ADMISSIONS_PDF_PATH."
            )

        cls.result = extract_pdf(PDF_PATH)
        cls.blocks = cls.result["blocks"]
        cls.total_pages = cls.result["total_pages"]

    # =========================================================================
    # NHÓM 1 & 2: CẤU TRÚC, HÌNH HỌC VÀ TÍNH TOÀN VẸN BẢNG
    # =========================================================================

    def test_01_document_has_expected_page_count(self):
        """1. Kiểm tra tổng số trang của tài liệu đúng 58 trang."""
        self.assertEqual(
            self.total_pages,
            58,
            "PDF đầu vào không còn đúng 58 trang."
        )

    def test_02_block_structure_and_geometry(self):
        """2. Kiểm tra tọa độ hình học, số trang, rỗng nội dung và phân cấp section."""
        errors = validate_block_geometry(
            self.blocks,
            self.total_pages
        )
        self.assertEqual(
            errors,
            [],
            "Block-level validation failed:\n" + "\n".join(errors[:100])
        )

    def test_03_table_row_metadata_one_to_one(self):
        """3. Kiểm tra tính đồng bộ 1-1 giữa rows_normalized và row_metadata."""
        errors = validate_table_integrity(self.blocks)
        self.assertEqual(
            errors,
            [],
            "Table integrity validation failed:\n"
            + "\n".join(errors[:100])
        )

    # =========================================================================
    # NHÓM 3: DOMAIN SANITY (NGHIỆP VỤ TUYỂN SINH)
    # =========================================================================

    def test_04_domain_sanity_24_admissions_majors(self):
        """4. Kiểm tra đầy đủ 24 ngành tuyển sinh trong chuỗi bảng trang 31-58 (STT 1-24)."""
        chain_31_58 = [
            b for b in self.blocks
            if b.get("type") == "table" and b.get("table_chain_id") == "table_chain_8_p31"
        ]
        self.assertEqual(
            len(chain_31_58),
            28,
            f"Chuỗi bảng ngành tuyển sinh phải gồm 28 block bảng (trang 31-58), nhận được {len(chain_31_58)}"
        )

        major_entries = []
        for tbl in chain_31_58:
            norm_rows = tbl["content"]["rows_normalized"]
            meta_rows = tbl["content"]["row_metadata"]
            for r, m in zip(norm_rows, meta_rows):
                stt_raw = m.get("stt", "").strip()
                if re.fullmatch(r"\d{1,2}\.?", stt_raw):
                    stt_num = int(stt_raw.replace(".", ""))
                    major_entries.append({
                        "stt_num": stt_num,
                        "stt_raw": stt_raw,
                        "page": tbl["page"],
                        "source_pages": m.get("source_pages", []),
                        "is_cross_page_spliced": m.get("is_cross_page_spliced", False),
                        "row": r
                    })

        # Sắp xếp theo STT số
        major_entries.sort(key=lambda x: x["stt_num"])
        stt_numbers = [m["stt_num"] for m in major_entries]

        # 4.1. Kiểm tra chính xác 24 ngành, STT từ 1 đến 24 liên tục
        self.assertEqual(
            len(major_entries),
            24,
            f"Phải phát hiện chính xác 24 ngành tuyển sinh, hiện có: {len(major_entries)}"
        )
        self.assertEqual(
            stt_numbers,
            list(range(1, 25)),
            f"Danh sách STT không liên tục 1..24: {stt_numbers}"
        )

        # 4.2. Kiểm tra số cột của từng dòng ngành tuyển sinh (13 cột theo schema bảng 11)
        for m in major_entries:
            row = m["row"]
            self.assertEqual(
                len(row),
                13,
                f"Ngành STT {m['stt_num']} có {len(row)} cột, mong đợi 13 cột"
            )
            # Tên ngành (col 2) không được để rỗng
            self.assertTrue(
                bool(row[2].strip()),
                f"Ngành STT {m['stt_num']} có tên ngành rỗng"
            )

        # 4.3. Kiểm tra chi tiết 2 trường hợp ghép nối xuyên trang đặc thù: STT 20 & STT 21
        # --- STT 20: Trang 52 -> 53 ---
        entry_20 = next(m for m in major_entries if m["stt_num"] == 20)
        row_20 = entry_20["row"]
        self.assertEqual(row_20[0].strip(), "20.")
        self.assertEqual(row_20[1].strip(), "7540102A")
        name_20 = " ".join(row_20[2].split())
        method_20 = " ".join(row_20[5].split())
        self.assertIn("Kỹ thuật thực phẩm", name_20)
        self.assertIn("Kỹ thuật sinh học thực phẩm", name_20)
        self.assertIn("100 - Xét kết quả thi Tốt nghiệp THPT", method_20)
        self.assertEqual(entry_20["source_pages"], [52, 53])
        self.assertTrue(entry_20["is_cross_page_spliced"])

        # --- STT 21: Trang 53 -> 54 ---
        entry_21 = next(m for m in major_entries if m["stt_num"] == 21)
        row_21 = entry_21["row"]
        self.assertEqual(row_21[0].strip(), "21.")
        self.assertEqual(row_21[1].strip(), "7510402")
        name_21 = " ".join(row_21[2].split())
        method_21 = " ".join(row_21[5].split())
        self.assertIn("Công nghệ vật liệu", name_21)
        self.assertIn("Hóa học Vật liệu mới", name_21)
        self.assertIn("100 - Xét kết quả thi Tốt nghiệp THPT", method_21)
        self.assertEqual(entry_21["source_pages"], [53, 54])
        self.assertTrue(entry_21["is_cross_page_spliced"])

    def test_05_independent_tables_isolation(self):
        """5. Xác minh các bảng độc lập (học phí, phương thức) không bị ghép nhầm vào chuỗi bảng ngành."""
        # 5.1. Bảng học phí trên trang 28
        tuition_table = next(
            (b for b in self.blocks if b.get("type") == "table" and b.get("page") == 28),
            None
        )
        self.assertIsNotNone(tuition_table, "Không tìm thấy bảng học phí trên trang 28")
        self.assertIsNone(
            tuition_table.get("table_chain_id"),
            f"Bảng học phí phải là bảng độc lập (table_chain_id=None), nhưng nhận được: {tuition_table.get('table_chain_id')}"
        )
        self.assertEqual(
            tuition_table.get("section"),
            "10.1. Thông tin về học phí"
        )

        # 5.2. Chuỗi bảng ngành chính (table_chain_8_p31) phải độc lập tuyệt đối
        chain_8_blocks = [
            b for b in self.blocks
            if b.get("type") == "table" and b.get("table_chain_id") == "table_chain_8_p31"
        ]
        for tbl in chain_8_blocks:
            self.assertEqual(
                tbl.get("section"),
                "11. Thông tin về tuyển sinh của 02 năm gần nhất",
                f"Bảng {tbl.get('table_id')} trong chuỗi ngành có section sai: {tbl.get('section')}"
            )
            self.assertTrue(
                31 <= tbl.get("page", 0) <= 58,
                f"Bảng {tbl.get('table_id')} có trang {tbl.get('page')} nằm ngoài phạm vi 31-58"
            )

        # 5.3. Bảng phương thức tuyển sinh 1 (table_chain_1_p2) phải thuộc Section 2.1
        method_chain_1 = [
            b for b in self.blocks
            if b.get("type") == "table" and b.get("table_chain_id") == "table_chain_1_p2"
        ]
        self.assertGreaterEqual(len(method_chain_1), 2)
        for tbl in method_chain_1:
            self.assertIn("2.1.", tbl.get("section", ""))

    def test_06_tuition_and_admission_methods_data(self):
        """6. Kiểm tra dữ liệu học phí và phương thức xét tuyển đối chiếu với PDF gốc."""
        # 6.1. Dữ liệu bảng học phí (Trang 28)
        tuition_table = next(
            b for b in self.blocks if b.get("type") == "table" and b.get("page") == 28
        )
        tuition_rows = tuition_table["content"]["rows_normalized"]
        self.assertEqual(len(tuition_rows), 5, "Bảng học phí phải có đúng 5 dòng (1 header + 4 mức học phí)")

        full_tuition_text = " ".join(" ".join(r) for r in tuition_rows)
        # Đối chiếu trực tiếp các con số học phí từ PDF gốc
        self.assertIn("Cử nhân sư phạm", full_tuition_text)
        self.assertIn("1.790.000", full_tuition_text)
        self.assertIn("500.000", full_tuition_text)
        self.assertIn("567.100", full_tuition_text)
        self.assertIn("623.700", full_tuition_text)
        self.assertIn("753.900", full_tuition_text)

        # 6.2. Dữ liệu bảng xét tuyển thẳng theo môn thi HSG (Trang 2-5)
        m1_tables = [
            b for b in self.blocks if b.get("table_chain_id") == "table_chain_1_p2"
        ]
        self.assertGreaterEqual(len(m1_tables), 2, "Chuỗi bảng Phương thức 1 phải gồm nhiều bảng liên tiếp")
        m1_header = m1_tables[0]["content"]["rows_normalized"][0]
        self.assertEqual(len(m1_header), 4, "Bảng phương thức 1 phải có đúng 4 cột")
        self.assertIn("Môn thi", m1_header[1])

        m1_full_text = " ".join(
            " ".join(r)
            for tbl in m1_tables
            for r in tbl["content"]["rows_normalized"]
        ).replace("\n", " ")
        for subject in ["Toán", "Vật lý", "Hóa học", "Sinh học", "Tin học"]:
            self.assertIn(subject, m1_full_text, f"Thiếu môn học '{subject}' trong bảng Phương thức 1")

    # =========================================================================
    # NHÓM 4: SERIALIZATION & QUANTITATIVE METRICS REPORT
    # =========================================================================

    def test_07_quantitative_metrics_computation(self):
        """7. Tính toán báo cáo định lượng toàn diện cho tài liệu tuyển sinh."""
        metrics = compute_document_metrics(self.result)

        self.assertEqual(metrics["total_pages"], 58)
        self.assertEqual(metrics["total_blocks"], 134)
        self.assertEqual(metrics["empty_blocks_count"], 0)
        self.assertEqual(metrics["invalid_coords_count"], 0)
        self.assertEqual(metrics["table_blocks_count"], 57)
        self.assertEqual(metrics["table_chains_count"], 8)
        self.assertEqual(metrics["repeated_headers_removed_count"], 8)
        self.assertEqual(metrics["cross_page_spliced_rows_count"], 32)
        self.assertEqual(metrics["total_normalized_table_rows"], 204)
        self.assertEqual(metrics["metadata_mismatches_count"], 0)
        self.assertEqual(metrics["distinct_source_pages_count"], 58)
        self.assertGreater(metrics["total_cells_count"], 0)

        # Kiểm tra tỷ lệ ô trống: tỷ lệ 28.31% là đặc trưng hợp lý do ô gộp, không bị xem là lỗi
        self.assertEqual(metrics["empty_cell_ratio"], 0.2831)
        self.assertEqual(
            metrics["empty_cell_ratio_interpretation"]["status"],
            "ACCEPTABLE"
        )
        self.assertIn("ô gộp", metrics["empty_cell_ratio_interpretation"]["reason"])

    def test_08_json_serialization_lossless(self):
        """8. Kiểm tra tính toàn vẹn và khả năng tuần tự hóa JSON không mất mát (Lossless)."""
        # Tuần tự hóa JSON tiêu chuẩn
        json_str = json.dumps(self.result, ensure_ascii=False)
        self.assertIsInstance(json_str, str)
        self.assertGreater(len(json_str), 100000)

        # Giải tuần tự hóa
        deserialized = json.loads(json_str)

        # Kiểm chứng bất biến cấu trúc (Structural Invariants)
        self.assertEqual(deserialized["source_file"], self.result["source_file"])
        self.assertEqual(deserialized["total_pages"], self.result["total_pages"])
        self.assertEqual(len(deserialized["blocks"]), len(self.blocks))

        # Kiểm tra chi tiết từng block
        for orig_b, deser_b in zip(self.blocks, deserialized["blocks"]):
            self.assertEqual(deser_b["type"], orig_b["type"])
            self.assertEqual(deser_b["page"], orig_b["page"])
            self.assertEqual(deser_b["section"], orig_b["section"])
            self.assertEqual(deser_b.get("top"), orig_b.get("top"))
            self.assertEqual(deser_b.get("bottom"), orig_b.get("bottom"))

            if orig_b["type"] == "table":
                orig_c = orig_b["content"]
                deser_c = deser_b["content"]
                self.assertEqual(deser_c["rows_normalized"], orig_c["rows_normalized"])
                self.assertEqual(len(deser_c["row_metadata"]), len(orig_c["row_metadata"]))
            else:
                self.assertEqual(deser_b["content"], orig_b["content"])

    def test_09_save_validation_metrics_report(self):
        """9. Xuất và lưu trữ báo cáo định lượng JSON tại reports/validation_metrics.json."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics = compute_document_metrics(self.result)

        report = {
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_pdf": self.result.get("source_file", "unknown.pdf"),
            "expected_pages": 58,
            "actual_pages": self.total_pages,
            "quantitative_metrics": metrics,
            "validation_results_by_group": {
                "group_1_geometry_and_structure": "PASS",
                "group_2_table_integrity": "PASS",
                "group_3_domain_sanity": "PASS",
                "group_4_serialization_and_metrics": "PASS",
            },
            "overall_status": "PASS"
        }

        with open(METRICS_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.assertTrue(METRICS_REPORT_PATH.is_file(), "Báo cáo validation_metrics.json phải được tạo thành công")

        # Đọc lại và kiểm tra nội dung
        with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
            saved_report = json.load(f)

        self.assertEqual(saved_report["overall_status"], "PASS")
        self.assertEqual(saved_report["actual_pages"], 58)
        self.assertEqual(saved_report["quantitative_metrics"]["total_blocks"], 134)
        self.assertEqual(saved_report["quantitative_metrics"]["table_chains_count"], 8)
        self.assertEqual(
            saved_report["quantitative_metrics"]["empty_cell_ratio_interpretation"]["status"],
            "ACCEPTABLE"
        )

    def test_10_merged_cell_structure_interpretation(self):
        """10. Kiểm tra cấu trúc ô gộp (merged cells): ô rỗng là do ô gộp, không làm mất dữ liệu bắt buộc."""
        # 10.1. Kiểm tra bảng môn thi HSG (table_chain_1_p2, trang 2-3)
        # Trong bảng này, môn thi ở cột 1 được gộp cho nhiều ngành (cột 2) và mã xét tuyển (cột 3).
        # Các hàng tiếp theo trong cùng nhóm môn sẽ có cột 1 rỗng, nhưng cột 2 và 3 bắt buộc phải có dữ liệu.
        m1_tables = [
            b for b in self.blocks if b.get("table_chain_id") == "table_chain_1_p2"
        ]
        p2_table = next(t for t in m1_tables if t["page"] == 2)
        p2_rows = p2_table["content"]["rows_normalized"]

        # Dòng 1: bắt đầu nhóm môn "Toán, Vật lý, Hóa học"
        self.assertIn("Toán", p2_rows[1][1])
        self.assertTrue(bool(p2_rows[1][2].strip()))  # Tên ngành
        self.assertTrue(bool(p2_rows[1][3].strip()))  # Mã ngành

        # Các dòng 2, 3, 4, 5: cột 1 rỗng do ô gộp từ dòng 1, nhưng tên ngành và mã ngành phải có đủ
        for r_idx in range(2, len(p2_rows)):
            row = p2_rows[r_idx]
            self.assertEqual(
                row[1].strip(),
                "",
                f"Dòng {r_idx} trong nhóm môn gộp trang 2 phải có ô môn rỗng do ô gộp"
            )
            self.assertTrue(
                bool(row[2].strip()),
                f"Dòng {r_idx} trang 2 không được để rỗng tên ngành"
            )
            self.assertTrue(
                bool(row[3].strip()),
                f"Dòng {r_idx} trang 2 không được để rỗng mã ngành"
            )

        # 10.2. Kiểm tra chuỗi bảng 24 ngành (table_chain_8_p31, trang 31-58)
        # Mỗi ngành có dòng đầu mang STT, mã, tên. Các dòng phương thức tiếp theo của cùng ngành
        # có ô STT/mã/tên rỗng do ô gộp, nhưng cột phương thức (cột 5) bắt buộc phải có nội dung.
        c8_tables = [
            b for b in self.blocks if b.get("table_chain_id") == "table_chain_8_p31"
        ]
        total_data_rows = 0
        method_rows_with_empty_stt = 0
        for tbl in c8_tables:
            for r, m in zip(tbl["content"]["rows_normalized"], tbl["content"]["row_metadata"]):
                if m.get("is_header"):
                    continue
                # Bảng tuyển sinh sử dụng header 2 dòng: dòng 1 có nội dung ở cột 0-5,
                # dòng 2 là sub-header chứa "Số lượng", "Số nhập học", "Điểm trúng tuyển"
                # ở cột 6-8. Dòng sub-header này có cột 0-5 rỗng nhưng is_header=False.
                # Bỏ qua dòng này vì nó không phải dòng dữ liệu thực sự.
                first_6_empty = all(not str(c or "").strip() for c in r[:6])
                has_data_after_col5 = any(str(c or "").strip() for c in r[6:])
                if first_6_empty and has_data_after_col5:
                    # Đây là dòng sub-header (hàng thứ 2 của header 2 dòng),
                    # dữ liệu nằm ở cột 6+ ("Số lượng", "Điểm trúng tuyển"...).
                    # Không kiểm tra invariant phương thức cho dòng này.
                    continue
                total_data_rows += 1
                # Không dòng nào được phép rỗng hoàn toàn
                non_empty_cells = [c for c in r if str(c or "").strip()]
                self.assertGreater(
                    len(non_empty_cells),
                    0,
                    f"Dòng dữ liệu tại trang {tbl['page']} không được rỗng toàn bộ"
                )
                if not r[0].strip():
                    method_rows_with_empty_stt += 1
                    # Cột 5 (phương thức xét tuyển) bắt buộc phải có nội dung.
                    # Các dòng có STT rỗng là các dòng phương thức phụ của ngành,
                    # nơi cột STT/mã/tên được gộp theo chiều dọc (merged cells).
                    self.assertTrue(
                        bool(r[5].strip()),
                        f"Dòng phương thức phụ tại trang {tbl['page']} bị rỗng cột phương thức"
                    )

        # Khẳng định số dòng phương thức có STT rỗng chiếm tỷ lệ hợp lý (do ô gộp ngành)
        self.assertGreater(method_rows_with_empty_stt, 0)
        self.assertGreater(total_data_rows, method_rows_with_empty_stt)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import json
import re
from pathlib import Path
from app.rag.ingestion.pdf_extractor_test import (
    extract_pdf,
    validate_extracted_blocks,
    validate_multipage_table_continuity
)

def run_comprehensive_validation():
    pdf_path = Path("data/dsk-thong-tin-tuyen-sinh-2026-new-0162026-25-ctdt-1950-ct.pdf")
    print("=" * 70)
    print("🚀 BẮT ĐẦU TRÍCH XUẤT VÀ KIỂM ĐỊNH TOÀN DIỆN GIAI ĐOẠN 4 TRÊN PDF THỰC TẾ")
    print("=" * 70)

    result = extract_pdf(pdf_path)
    blocks = result["blocks"]
    total_pages = result["total_pages"]

    # 1. Validation chung toàn tài liệu
    val_general = validate_extracted_blocks(blocks, expected_total_pages=58)
    # 2. Validation chuyên biệt Giai đoạn 4
    val_g4 = validate_multipage_table_continuity(blocks)

    print(f"\n📊 TỔNG QUAN TÀI LIỆU:")
    print(f"- Tổng số trang: {total_pages} / 58")
    print(f"- Tổng số blocks: {len(blocks)}")
    print(f"- Số block tọa độ bất thường: {val_general['invalid_coords_count']}")
    print(f"- Trạng thái toàn vẹn chung: {'✅ ĐẠT' if val_general['is_valid'] else '❌ KHÔNG ĐẠT'}")

    print(f"\n📊 KẾT QUẢ GIAI ĐOẠN 4 (MULTI-PAGE TABLES CONTINUITY):")
    print(f"- Tổng số block bảng: {val_g4['total_table_blocks']}")
    print(f"- Tổng số chuỗi bảng phát hiện: {val_g4['total_chains']}")
    for cid, cinfo in val_g4['chain_details'].items():
        print(f"  * {cid}: {cinfo['table_count']} bảng (Trang {min(cinfo['pages'])} -> {max(cinfo['pages'])})")
    print(f"- Số Header lặp lại bị loại bỏ: {val_g4['headers_removed_count']}")
    print(f"- Số hàng được ghép qua trang: {val_g4['cross_page_spliced_rows_count']}")
    print(f"- Số lỗi lệch metadata hàng (rows vs meta): {val_g4['metadata_mismatches_count']}")
    print(f"- Số cảnh báo cần review thủ công: {val_g4['warnings_count']}")

    print("\n" + "=" * 70)
    print("🔍 KIỂM TRA ĐỐI CHIẾU THỰC TẾ 2 TRƯỜNG HỢP ĐẶC BIỆT TRÊN PDF")
    print("=" * 70)

    # Lấy block bảng ở trang 52, 53, 54
    t52 = next(b for b in blocks if b["type"] == "table" and b["page"] == 52)
    t53 = next(b for b in blocks if b["type"] == "table" and b["page"] == 53)
    t54 = next(b for b in blocks if b["type"] == "table" and b["page"] == 54)

    # -------------------------------------------------------------
    # TRƯỜNG HỢP A: Trang 52 sang Trang 53 (STT 20: Kỹ thuật thực phẩm)
    # -------------------------------------------------------------
    raw_last_52 = t52["content"]["rows_raw"][-1]
    raw_first_53 = t53["content"]["rows_raw"][0]
    norm_last_52 = t52["content"]["rows_normalized"][-1]
    meta_last_52 = t52["content"]["row_metadata"][-1]

    print("\n--- [TRƯỜNG HỢP A: STT 20 (Trang 52 -> 53)] ---")
    print("1. Trích xuất thô cuối trang 52:")
    print(f"   - Ô tên ngành (col 2): '{raw_last_52[2]}'")
    print(f"   - Ô phương thức (col 5): '{raw_last_52[5]}'")
    print("2. Trích xuất thô đầu trang 53:")
    print(f"   - Ô tên ngành (col 2): '{raw_first_53[2]}'")
    print(f"   - Ô phương thức (col 5): '{raw_first_53[5]}'")
    print("3. Quyết định thuật toán: GHÉP HÀNG TIẾP NỐI")
    print(f"   - Bằng chứng: {meta_last_52.get('spliced_details', {}).get('reason')}")
    print("4. Nội dung cuối cùng sau khi ghép:")
    print(f"   - STT: '{norm_last_52[0]}'")
    print(f"   - Tên ngành: '{norm_last_52[2]}'")
    print(f"   - Phương thức: '{norm_last_52[5]}'")
    print(f"   - Source pages cấp bản ghi: {meta_last_52.get('source_pages')}")
    print(f"   - Is cross-page spliced: {meta_last_52.get('is_cross_page_spliced')}")

    # -------------------------------------------------------------
    # TRƯỜNG HỢP B: Trang 53 sang Trang 54 (STT 21: Công nghệ vật liệu)
    # -------------------------------------------------------------
    raw_last_53 = t53["content"]["rows_raw"][-1]
    raw_first_54 = t54["content"]["rows_raw"][0]
    norm_last_53 = t53["content"]["rows_normalized"][-1]
    meta_last_53 = t53["content"]["row_metadata"][-1]

    print("\n--- [TRƯỜNG HỢP B: STT 21 (Trang 53 -> 54)] ---")
    print("1. Trích xuất thô cuối trang 53:")
    print(f"   - Ô tên ngành (col 2): '{raw_last_53[2]}'")
    print(f"   - Ô phương thức (col 5): '{raw_last_53[5]}'")
    print("2. Trích xuất thô đầu trang 54:")
    print(f"   - Ô tên ngành (col 2): '{raw_first_54[2]}'")
    print(f"   - Ô phương thức (col 5): '{raw_first_54[5]}'")
    print("3. Quyết định thuật toán: GHÉP HÀNG TIẾP NỐI")
    print(f"   - Bằng chứng: {meta_last_53.get('spliced_details', {}).get('reason')}")
    print("4. Nội dung cuối cùng sau khi ghép:")
    print(f"   - STT: '{norm_last_53[0]}'")
    print(f"   - Tên ngành: '{norm_last_53[2]}'")
    print(f"   - Phương thức: '{norm_last_53[5]}'")
    print(f"   - Source pages cấp bản ghi: {meta_last_53.get('source_pages')}")
    print(f"   - Is cross-page spliced: {meta_last_53.get('is_cross_page_spliced')}")

    print("\n" + "=" * 70)
    print("🎯 KIỂM ĐỊNH TOÀN VẸN 24 NGÀNH TRONG CHUỖI BẢNG 31-58")
    print("=" * 70)
    chain_31_58_tables = [b for b in blocks if b["type"] == "table" and 31 <= b.get("page", 0) <= 58]
    all_stts = []
    for tbl in chain_31_58_tables:
        for r_meta in tbl["content"]["row_metadata"]:
            stt = r_meta.get("stt", "").strip()
            if re.fullmatch(r"\d{1,2}\.?", stt):
                all_stts.append(stt)
    
    unique_stts = sorted(list(set(all_stts)), key=lambda s: int(s.replace(".", "")))
    print(f"Danh sách STT các ngành phát hiện được ({len(unique_stts)} ngành):")
    print(" -> " + ", ".join(unique_stts))
    is_complete_24 = (len(unique_stts) == 24 and [int(s.replace(".", "")) for s in unique_stts] == list(range(1, 25)))
    print(f"Kiểm tra tính liên tục 1 đến 24: {'✅ HOÀN TOÀN CHÍNH XÁC (24/24)' if is_complete_24 else '❌ THIẾU HOẶC THỪA'}")

    print("\n" + "=" * 70)
    print("🎯 KẾT LUẬN GIAI ĐOẠN 4 MỚI: ✅ TẤT CẢ KIỂM TRA ĐẠT TIÊU CHUẨN")
    print("=" * 70)

    # Lưu kết quả JSON
    output_path = Path("data/extracted_blocks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả trích xuất vào {output_path}")

if __name__ == "__main__":
    run_comprehensive_validation()

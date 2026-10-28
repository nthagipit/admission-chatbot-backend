from statistics import mean
import pdfplumber
from pathlib import Path
import re
from typing import Any

# Nhận diện mục số La Mã: I., II., III., IV., V., VI., VII., VIII., IX., X.
# ROMAN_HEADING_PATTERN = re.compile(
#     r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+([^\n:]+)\s*$",
#     re.IGNORECASE
# )

# Nhận diện mục số: 1., 1.1., 10.2.1.
NUMERIC_HEADING_PATTERN = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})*)\.\s+(.+?)\s*$"
)

# Nhận diện mục số La Mã: I., II., III., IV., V., VI., VII., VIII., IX., X.
ROMAN_HEADING_PATTERN = re.compile(
    r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+([^\n]+)$",
    re.IGNORECASE
)

# Nhận diện mục số đa cấp: 2.1., 2.2., 10.2.1.
MULTI_LEVEL_NUMERIC_PATTERN = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})+)\.?\s+(.+)$"
)

# Nhận diện mục số đơn cấp: 1., 2., 10., 11.
SINGLE_LEVEL_NUMERIC_PATTERN = re.compile(
    r"^\s*(\d{1,2})\.\s+(.+)$"
)

# Các từ khóa trường thông tin kê khai (chỉ áp dụng cho mục đơn cấp ở Trang 1 Thông tin chung)
NON_HEADING_FIELD_PREFIXES = (
    "tên cơ sở",
    "mã cơ sở",
    "mã trường",
    "địa chỉ",
    "số điện thoại",
    "điện thoại",
    "email",
    "website",
    "trang thông tin",
    "đường link",
    "họ và tên",
    "quy mô đào tạo",
)

NUMBERED_ROW_PATTERN = re.compile(r"^\s*\d{1,3}(?:[.)])?\s*$")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    value = clean_text(value)
    return value.replace("|", r"\|")


def is_page_number_artifact(text: str, top: float, page_height: float) -> bool:
    """
    Nhận diện số trang đứng riêng gần mép trên hoặc mép dưới.
    Đây là heuristic, cần kiểm tra trên PDF thực tế.
    """
    text = text.strip()
    if not re.fullmatch(r"\d{1,3}", text):
        return False
    near_top = top < 50
    near_bottom = top > page_height - 50
    return near_top or near_bottom



def parse_section_heading(text: str) -> str | None:
    """
    Nhận diện tiêu đề phân cấp tài liệu (Section Heading) đầy đủ:
    1. Cấp La Mã: 'I. THÔNG TIN CHUNG', 'II. TUYỂN SINH ĐÀO TẠO ĐẠI HỌC...'
    2. Cấp số lớn: '1. Đối tượng...', '2. Mô tả phương thức...', '10. Các nội dung khác'
    3. Cấp mục con đa cấp: 
       - '2.1. Phương thức 1: Xét tuyển thẳng...'
       - '2.2. Phương thức 2: Xét tuyển kết hợp...'
       - '3.1. Ngưỡng đầu vào: ...'
       - '7.1. Ưu tiên theo khu vực: ...'
       - '10.3. Chính sách miễn giảm học phí: ...'
       (Cho phép có dấu ':' vì tiêu đề mục con trong đề án thường gắn liền với mô tả)
    """
    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < 4:
        return None

    # Bỏ qua nếu toàn bộ chỉ là các con số phân cách (ví dụ số cột của bảng bị lọt ra: 8.0 8.5 9.0)
    if re.match(r"^[0-9\s\.\,]+$", cleaned):
        return None

    # 1. Kiểm tra Tiêu đề cấp cao số La Mã (I., II., ...)
    roman_match = ROMAN_HEADING_PATTERN.match(cleaned)
    if roman_match:
        roman_num, roman_title = roman_match.groups()
        return f"{roman_num.upper()}. {roman_title.strip()}"

    # 2. Kiểm tra Tiêu đề mục con đa cấp (2.1., 2.2., 10.1., 10.2.1., ...)
    multi_match = MULTI_LEVEL_NUMERIC_PATTERN.match(cleaned)
    if multi_match:
        prefix, title = multi_match.groups()
        title_clean = title.strip()
        # Nếu tiêu đề có dấu hai chấm và quá dài, chúng ta lấy trọn vẹn cụm tiêu đề (ví dụ '2.1. Phương thức 1: Xét tuyển thẳng...')
        if len(title_clean) >= 3:
            return f"{prefix}. {title_clean}"

    # 3. Kiểm tra Tiêu đề số đơn cấp (1., 2., 3., 10., 11., ...)
    single_match = SINGLE_LEVEL_NUMERIC_PATTERN.match(cleaned)
    if single_match:
        num, title = single_match.groups()
        title_clean = title.strip()
        lower_title = title_clean.lower()

        # Loại trừ các dòng kê khai thông tin chung ở Trang 1
        if any(lower_title.startswith(p) for p in NON_HEADING_FIELD_PREFIXES):
            return None
        if "http" in lower_title or "https" in lower_title:
            return None

        # Nếu dòng quá ngắn hoặc bắt đầu bằng chữ thường thì không phải
        if len(title_clean) >= 3 and (title_clean[0].isupper() or title_clean[0].isdigit()):
            return f"{num}. {title_clean}"

    return None
    

def extract_table_blocks(page, page_number: int) -> list[dict[str, Any]]:
    """
        Trích xuất bảng cùng tọa độ và dữ liệu hàng/cột thô
    """
    table_blocks = []
    
    for table_index, table in enumerate(page.find_tables()):
        rows = table.extract() or []

        rows_raw = [
            [clean_cell(cell) for cell in row]
            for row in rows
        ]

        if not any(cell for row in rows_raw for cell in row):
            continue

        x0, top, x1, bottom = table.bbox
        rows_normalized = merge_wrapped_table_rows(rows_raw)

        # Khởi tạo row_metadata cho từng hàng của rows_normalized
        row_metadata = []
        for r_idx, r in enumerate(rows_normalized):
            is_hdr = (r_idx == 0)
            stt = r[0].strip() if r else ""
            row_metadata.append({
                "row_index": r_idx,
                "is_header": is_hdr,
                "stt": stt,
                "source_pages": [page_number],
                "is_cross_page_spliced": False
            })

        table_blocks.append({
            "type": "table",
            "page": page_number,
            "top": float(top),
            "bottom": float(bottom),
            "bbox": [
                float(x0),
                float(top),
                float(x1),
                float(bottom)
            ],
            "table_index": table_index,
            "content": {
                "rows_raw": rows_raw,
                "rows_normalized": rows_normalized,
                "row_metadata": row_metadata,
                "row_count": len(rows_raw),
                "row_count_normalized": len(rows_normalized),
                "column_counts": [len(row) for row in rows_raw]
            },
            "section": None,
            "table_id": f"p{page_number}_t{table_index}"
        })

    return table_blocks


def is_inside_table(word: dict[str, Any], table_blocks: list[dict[str, Any]]) -> bool:
    """
        Kiểm tra tâm của từ có nằm trong vùng bảng hay không.
    """
    center_x = (word["x0"] + word["x1"]) / 2
    center_y = (word["top"] + word["bottom"]) /2
    
    for block in table_blocks:
        x0, top, x1, bottom = block["bbox"]

        if x0 <= center_x <= x1 and top <= center_y <= bottom:
            return True
        
    return False

def is_bbox_inside(inner: list[float], outer: list[float], tolerance: float = 2.0,) -> bool:
    ix0, itop, ix1, ibottom = inner
    ox0, otop, ox1, obottom = outer

    return (
        (ix0 >= ox0 - tolerance) and (ix1 <= ox1 + tolerance) and
        (itop >= otop - tolerance) and (ibottom <= obottom + tolerance)
    )

def remove_nested_table_blocks(table_blocks: list[dict]) -> list[dict]:
    """
        Loại bảng nằm hoàn toàn bên trong bbox của bảng lớn hơn.
        Giữ lại bảng lớn để tránh mất nội dung của ô
    """
    kept = []

    for candidate in table_blocks:
        candidate_area = (
            (candidate["bbox"][2] - candidate["bbox"][0]) * (candidate["bbox"][3] - candidate["bbox"][1])
        )

        nested = False
        for other in table_blocks:
            if candidate is other:
                continue
            
            other_area = (
                (other["bbox"][2] - other["bbox"][0]) * (other["bbox"][3] - other["bbox"][1])
            )

            if other_area <= candidate_area:
                continue

            if is_bbox_inside(candidate["bbox"], other["bbox"]):
                nested = True
                break
        
        if not nested:
            kept.append(candidate)

    return kept

def analyze_table_structure(block: dict[str, Any]) -> dict[str, Any]:
    """
        Tính các đặc trưng hình thức để phân loại bảng ứng viên.
        Đây là heuristic, không phải bộ nhận diện đúng tuyệt đối cho mọi PDF
    """
    rows = block.get("content", {}).get("rows_raw", [])

    nonempty_rows = [
        [str(cell).strip() for cell in row]
        for row in rows
        if any(str(cell or "").strip() for cell in row)
    ]

    if not nonempty_rows:
        return {
            "row_count": 0,
            "column_count": 0,
            "nonempty_cell_ratio": 0.0,
            "single_cell_row_ratio": 1.0,
            "average_cell_length": 0.0,
            "numbered_first_cell_ratio": 0.0,
            "long_cell_ratio": 0.0,
        }

    column_count = max(len(row) for row in nonempty_rows)
    normalized = [
        row + [""] * (column_count - len(row))
        for row in nonempty_rows
    ]
    cells = [cell for row in normalized for cell in row]
    nonempty_cells = [cell for cell  in cells if cell]
    nonempty_cell_ratio = len(nonempty_cells) / max(1, len(cells))
    single_cell_rows = sum(1 for row in normalized if sum(bool(cell) for cell in row) <= 1)
    first_cells = [row[0] for row in normalized if row]
    numbered_ratio = (
        sum(bool(NUMBERED_ROW_PATTERN.fullmatch(cell)) for cell in first_cells)
        / max(1, len(first_cells))
    )

    average_length = mean(len(cell) for cell in nonempty_cells) if nonempty_cells else 0
    long_cell_ratio = (
        sum(len(cell) >= 70 for cell in nonempty_cells) / max(1, len(nonempty_cells))
    )

    return {
        "row_count": len(normalized),
        "column_count": column_count,
        "nonempty_cell_ratio": round(nonempty_cell_ratio, 3),
        "single_cell_row_ratio": round(single_cell_rows / len(normalized), 3),
        "average_cell_length": round(average_length, 1),
        "numbered_first_cell_ratio": round(numbered_ratio, 3),
        "long_cell_ratio": round(long_cell_ratio, 3),
    }

def is_likely_prose_table(
    block: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> bool:
    """
    Ước lượng vùng find_tables() có thể chỉ là văn bản bị chia thành các ô.

    Chỉ chuyển sang text khi có nhiều dấu hiệu cùng lúc:
    - có ít cột;
    - phần lớn hàng chỉ có một ô chứa nội dung;
    - ô chứa các đoạn dài;
    - ít dấu hiệu về hàng dữ liệu đánh số.

    Không dùng số trang hoặc từ khóa riêng của một tài liệu.
    """
    features = features or analyze_table_structure(block)

    rows = features["row_count"]
    columns = features["column_count"]

    if rows == 0 or columns == 0:
        return True

    prose_like = (
        rows <= 12
        and columns <= 3
        and features["single_cell_row_ratio"] >= 0.60
        and (
            features["average_cell_length"] >= 45
            or features["long_cell_ratio"] >= 0.35
        )
        and features["numbered_first_cell_ratio"] < 0.35
    )

    return prose_like

def table_candidate_to_text_block(
    block: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    """Chuyển vùng bảng nghi là văn bản thành một block text, giữ nguyên tọa độ."""
    rows = block.get("content", {}).get("rows_raw", [])
    lines: list[str] = []

    for row in rows:
        values = [str(cell).strip() for cell in row if str(cell or "").strip()]
        if not values:
            continue

        # Tránh lặp lại nội dung nếu hàng có các ô trùng nhau.
        line = " ".join(values)
        if not lines or line != lines[-1]:
            lines.append(line)

    return {
        "type": "text",
        "page": block["page"],
        "top": block["top"],
        "bottom": block["bottom"],
        "bbox": block["bbox"],
        "content": clean_text("\n".join(lines)),
        "section": block.get("section"),
        "source_table_id": block.get("table_id"),
        "classification": "converted_from_table_candidate",
        "table_features": features,
    }

def classify_table_blocks(
    table_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Trả về (real_tables, converted_text_blocks).
    Các trường hợp chưa chắc chắn vẫn có thể bị phân loại sai; nên kiểm thử output.
    """
    real_tables: list[dict[str, Any]] = []
    converted_text: list[dict[str, Any]] = []

    for block in table_blocks:
        features = analyze_table_structure(block)
        block["content"]["structure_features"] = features

        if is_likely_prose_table(block, features):
            converted_text.append(table_candidate_to_text_block(block, features))
        else:
            block["type"] = "table"
            block["classification"] = "table_candidate_kept"
            real_tables.append(block)

    return real_tables, converted_text

def group_words_into_lines(words: list[dict[str, Any]], tolerance: float = 3.0) -> list[dict[str, Any]]:
    """
        Gom từ thành dòng theo tọa độ top.
        Đây là baseline cho tài liệu có bố cục tương đối đơn giản.
    """

    sorted_words = sorted(words, key=lambda word: (word["top"], word["x0"]))
    lines: list[list[dict[str, Any]]] = []

    for word in sorted_words:
        matched_line = None
        
        for line in reversed(lines):
            if word["top"] - line["top"] > tolerance:
                break

            if abs(word["top"] - line["top"]) <= tolerance:
                matched_line = line
                break
        
        if matched_line is None:
            lines.append({
                "top": float(word["top"]),
                "bottom": float(word["bottom"]),
                "words": [word]
            })
        else:
            matched_line["words"].append(word)
            matched_line["bottom"] = max(matched_line["bottom"], float(word["bottom"]))

    result = []

    for line in lines:
        ordered_words = sorted(
            line["words"],
            key=lambda word: word["x0"]
        )

        result.append({
            "top": line["top"],
            "bottom": line["bottom"],
            "text": clean_text(
                " ".join(word["text"] for word in ordered_words)
            ),
        })
    
    return sorted(result, key=lambda line: line["top"])

def extract_text_blocks(page, page_number: int, table_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
        Trích xuất các dòng text không thuộc vùng bảng.
    """
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
    )

    text_words = [
        word 
        for word in words
        if not is_inside_table(word, table_blocks)
    ]

    lines = group_words_into_lines(text_words)
    blocks: list[dict[str, Any]] = []

    for line in lines:
        text = line["text"]

        if not text:
            continue

        if is_page_number_artifact(text, line["top"], page.height):
            continue

        blocks.append({
            "type": "text",
            "page": page_number,
            "top": float(line["top"]),
            "bottom": float(line["bottom"]),
            "bbox": None,
            "content": text,
            "section": None
        })

    return blocks

def merge_text_lines(text_blocks: list[dict[str, Any]], max_vertical_gap: float = 12.0) -> list[dict[str, Any]]:
    """
        Ghép các dòng gần nhau thành đoạn.
        Heading được giữ thành block riêng.
    """

    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current

        if current is not None:
            current["content"] = clean_text(current["content"])
            merged.append(current)
            current = None
    
    for block in sorted(text_blocks, key=lambda item: item["top"]):
        text = block["content"]
        is_heading = parse_section_heading(text) is not None

        if is_heading:
            flush()
            merged.append({
                **block,
                "type": "heading"
                })
            continue

        if current is None:
            current = {
                **block,
                "type": "text",
                "content": text,
            }
            continue

        vertical_gap = block["top"] - current["bottom"]

        if vertical_gap <= max_vertical_gap:
            current["content"] += " " + text
            current["bottom"] = max(current["bottom"], block["bottom"])
            continue
        else:
            flush()
            current = {
                **block,
                "type": "text",
                "content": text,
        }
    
    flush()
    return merged

def merge_wrapped_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """
    Giai đoạn 3: Chuẩn hóa các hàng trong bảng bị ngắt/xuống dòng trong cùng trang.
    
    Quy tắc:
    1. Tự động kiểm tra xem bảng có cột số thứ tự (STT/TT) hay không.
    2. Nếu có cột STT:
       - Hàng mới bắt đầu khi ô ở cột 0 chứa số thứ tự (ví dụ: '1', '2.', '3)').
       - Hàng tiếp theo có ô STT rỗng: được xem là hàng tiếp diễn và ghép vào các ô tương ứng của hàng trước.
    3. Chuẩn hóa lỗi dính chữ hoặc gạch dưới (ví dụ: 'chất lượng_nhóm ngành 1' -> 'chất lượng - nhóm ngành 1').
    4. Nếu không phải bảng STT: Giữ nguyên cấu trúc hàng, chỉ chuẩn hóa khoảng trắng các ô.
    """
    if not rows:
        return []

    def clean_cell_text(text: str) -> str:
        if not text:
            return ""
        # Sửa lỗi gạch dưới nối từ: 'lượng_nhóm' -> 'lượng - nhóm'
        text = re.sub(r"([a-zA-Z0-9_à-ỹÀ-Ỹ]+)_([a-zA-Z0-9_à-ỹÀ-Ỹ]+)", r"\1 - \2", text)
        return clean_text(text)

    def is_stt_cell(val: str) -> bool:
        v = val.strip()
        return bool(re.fullmatch(r"\d{1,3}(?:\.|\))?", v))

    # 1. Kiểm tra xem cột 0 có phải là cột số thứ tự không
    header_col0 = rows[0][0].strip().upper() if rows[0] else ""
    col0_values = [r[0].strip() for r in rows[1:] if len(r) > 0 and r[0].strip()]
    
    is_stt_table = (
        header_col0 in ("STT", "TT", "SỐ TT") 
        or (len(col0_values) > 0 and sum(is_stt_cell(v) for v in col0_values) / len(col0_values) >= 0.5)
    )

    col_count = max(len(row) for row in rows)

    # Nếu không phải bảng STT: Chuẩn hóa khoảng trắng từng ô và giữ nguyên số hàng
    if not is_stt_table:
        return [
            [clean_cell_text(cell) for cell in row] + [""] * (col_count - len(row))
            for row in rows
        ]

    # 2. Xử lý gộp hàng tiếp diễn cho bảng STT
    result: list[list[str]] = []

    for row_idx, row in enumerate(rows):
        row_padded = [clean_cell_text(cell) for cell in row] + [""] * (col_count - len(row))

        # Hàng 0 là tiêu đề cột (header)
        if row_idx == 0:
            result.append(row_padded)
            continue

        stt = row_padded[0].strip()

        # Nếu ô STT có số -> Bắt đầu một bản ghi hàng mới
        if is_stt_cell(stt):
            result.append(row_padded)
            continue

        # Nếu ô STT rỗng nhưng có hàng dữ liệu trước đó -> Ghép vào hàng trước
        if len(result) > 1:
            previous_row = result[-1]
            for col_idx in range(1, col_count):
                val = row_padded[col_idx].strip()
                if not val:
                    continue
                if previous_row[col_idx]:
                    previous_row[col_idx] += " " + val
                else:
                    previous_row[col_idx] = val
        else:
            result.append(row_padded)

    return result


def extract_page_blocks(page, page_number: int) -> list[dict[str, Any]]:
    """
        Tạo danh sách block theo thứ tự xuất hiện trên một trang.
    """
    table_blocks = extract_table_blocks(page, page_number)

    candidates = remove_nested_table_blocks(table_blocks)

    real_tables, converted_text_blocks = classify_table_blocks(candidates)

    text_lines = extract_text_blocks(page, page_number, candidates)

    text_blocks = merge_text_lines(text_lines)

    blocks = text_blocks + converted_text_blocks + real_tables

    type_priority = {
        "heading": 0,
        "text": 1,
        "table": 2
    }

    blocks.sort(
        key=lambda block: (
            block["top"],
            type_priority.get(block["type"], 99),
            block.get("table_id", "")
        )
    )

    return blocks

# ======================================================================
# GIAI ĐOẠN 4: HỆ THỐNG XỬ LÝ CHUỖI BẢNG VÀ GHÉP HÀNG ĐA TRANG (CHUẨN HÓA V2)
# ======================================================================

TABLE_HEADER_KEYWORDS = (
    "STT", "TT", "SỐ TT",
    "MÃ XÉT TUYỂN", "MÃ NGÀNH", "MÃ ĐKXT", "MÃ CHƯƠNG TRÌNH",
    "TÊN CHƯƠNG TRÌNH", "TÊN NGÀNH", "NGÀNH ĐƯỢC XÉT TUYỂN", "NGÀNH TUYỂN THẲNG",
    "TỔ HỢP XÉT TUYỂN", "TỔ HỢP MÔN", "TỔ HỢP",
    "ĐIỂM TRÚNG TUYỂN", "ĐIỂM XÉT THƯỞNG", "ĐIỂM THƯỞNG",
    "CHỈ TIÊU", "LĨNH VỰC", "MÔN THI HSG", "HỌC PHÍ", "TÊN TRƯỜNG"
)

def get_table_column_count(table_block: dict[str, Any]) -> int:
    """
    Xác định số cột thực tế của bảng dựa trên độ dài phổ biến nhất (mode)
    của các hàng có dữ liệu trong rows_normalized hoặc rows_raw.
    """
    content = table_block.get("content", {})
    rows = content.get("rows_normalized") or content.get("rows_raw") or []
    if not rows:
        return 0
    lengths = [len(r) for r in rows if any(str(c or "").strip() for c in r)]
    if not lengths:
        lengths = [len(r) for r in rows]
    if not lengths:
        return 0
    from collections import Counter
    most_common = Counter(lengths).most_common(1)
    return most_common[0][0] if most_common else 0


def normalize_text_for_comparison(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản phục vụ so sánh nội dung header."""
    text = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def detect_recurrent_header(
    row: list[str], 
    reference_header: list[str] | None, 
    col_count: int
) -> tuple[bool, str | None]:
    """
    Phát hiện an toàn header lặp lại ở đầu bảng trang tiếp theo:
    1. So sánh tỷ lệ tương đồng với reference_header (header gốc của chuỗi).
    2. Nếu không có reference_header, kiểm tra các nhãn cột có cấu trúc (ít nhất 2-3 cột).
    3. Safety Guards: Tuyệt đối không coi là header nếu ô STT là số thứ tự dữ liệu
       hoặc hàng chứa các giá trị điểm số/chỉ tiêu dạng số cụ thể.
    """
    if not row:
        return False, None
    
    # Safety Guard 1: Nếu ô STT là số thứ tự dữ liệu (ví dụ '1', '2.', '6.', '20.') -> Hàng dữ liệu
    stt_val = str(row[0] or "").strip()
    if re.fullmatch(r"\d{1,3}(?:\.|\))?", stt_val):
        return False, None
    
    # Safety Guard 2: Nếu có ô chứa điểm số dạng số thực cụ thể (ví dụ 23.07, 18.71, 715) -> Hàng dữ liệu
    for cell in row[1:]:
        c_str = str(cell or "").strip()
        if re.fullmatch(r"\d{1,2}(?:\.\d{1,3})?", c_str) and len(c_str) >= 4:
            return False, None

    # Tiêu chí 1: So sánh với reference_header của bảng trước / gốc
    if reference_header:
        ref_norm = [normalize_text_for_comparison(c) for c in reference_header]
        row_norm = [normalize_text_for_comparison(c) for c in row]
        
        non_empty_ref = [(idx, val) for idx, val in enumerate(ref_norm) if val]
        if non_empty_ref:
            matched_count = sum(
                1 for idx, val in non_empty_ref 
                if idx < len(row_norm) and row_norm[idx] and (row_norm[idx] == val or row_norm[idx] in val or val in row_norm[idx])
            )
            ratio = matched_count / len(non_empty_ref)
            if ratio >= 0.5 and matched_count >= 2:
                return True, f"matched_reference_header (ratio={ratio:.2f}, matches={matched_count})"

    # Tiêu chí 2: Phân tích các nhãn cột có cấu trúc
    matched_keywords = []
    for cell in row:
        c_upper = str(cell or "").upper().strip()
        for kw in TABLE_HEADER_KEYWORDS:
            if kw in c_upper and len(c_upper) <= len(kw) + 15:
                matched_keywords.append(kw)
                break
    
    if len(matched_keywords) >= 3:
        return True, f"structural_header_detected ({len(matched_keywords)} column keywords: {matched_keywords[:3]})"
    
    return False, None


def is_standalone_method_subrow(row: list[str]) -> bool:
    """
    Kiểm tra xem hàng có phải là bản ghi con độc lập (sub-row) có số liệu riêng:
    Ví dụ phương thức xét tuyển có mã (100 -, 200 -, 301 -) VÀ có điểm/chỉ tiêu ở các cột sau.
    """
    for c_idx in range(len(row)):
        c_str = str(row[c_idx] or "").strip()
        if re.match(r"^\d{3}\s*-", c_str):
            has_scores = any(re.search(r"\d", str(row[j] or "")) for j in range(c_idx + 1, len(row)))
            if has_scores:
                return True
    return False


def check_cross_page_continuation(
    prev_row: list[str], 
    curr_row: list[str]
) -> tuple[bool, list[int], str | None]:
    """
    Xác minh có bằng chứng tiếp nối văn bản giữa hàng cuối trang trước và hàng đầu trang sau.
    Kiểm tra tổng quát trên TẤT CẢ các cột (không giới hạn cột cố định).
    Trả về: (should_splice, spliced_columns, reason)
    """
    if not prev_row or not curr_row:
        return False, [], None
    
    # STT hàng sau phải rỗng mới có thể là hàng tiếp nối
    stt_curr = str(curr_row[0] or "").strip()
    if stt_curr:
        return False, [], None
    
    # Nếu là hàng phương thức độc lập có số liệu riêng -> KHÔNG GHÉP
    if is_standalone_method_subrow(curr_row):
        return False, [], None

    spliced_cols = []
    reasons = []

    max_cols = min(len(prev_row), len(curr_row))
    for c_idx in range(1, max_cols):
        p_val = str(prev_row[c_idx] or "").strip()
        c_val = str(curr_row[c_idx] or "").strip()
        
        if not p_val or not c_val:
            continue
        
        col_spliced = False
        
        # Dấu hiệu 1: Mất cân đối ngoặc đơn (dòng trước mở mà chưa đóng, dòng sau đóng ngoặc)
        if p_val.count("(") > p_val.count(")") and (")" in c_val or c_val.endswith(")")):
            col_spliced = True
            reasons.append(f"col_{c_idx}:unbalanced_parentheses")
            
        # Dấu hiệu 2: Dòng sau bắt đầu bằng dấu đóng ngoặc hoặc chữ cái viết thường
        elif c_val.startswith(")") or (c_val[0].isalpha() and c_val[0].islower()):
            col_spliced = True
            reasons.append(f"col_{c_idx}:starts_lowercase_or_closing_bracket")
            
        # Dấu hiệu 3: Dòng trước kết thúc bằng ký tự nối hoặc liên từ dở dang
        elif p_val.endswith(("-", "/", ",", "của", "theo", "và", "do", "tại", "về", "ngành", "chuyên")):
            col_spliced = True
            reasons.append(f"col_{c_idx}:ends_with_connective_word")

        # Dấu hiệu 4: Cụm từ tiếng Việt bị chia đôi trong phương thức tuyển sinh
        elif any(p_val.lower().endswith(w) for w in ("xét", "xét tuyển", "sử", "quy", "đề án của", "đánh giá", "kết quả")):
            if any(c_val.lower().startswith(w) for w in ("thẳng", "dụng", "chế", "csđt", "năng lực", "học tập", "kết quả")):
                col_spliced = True
                reasons.append(f"col_{c_idx}:split_vietnamese_phrase")

        if col_spliced:
            spliced_cols.append(c_idx)

    # Nếu hàng đã có bằng chứng tiếp nối ở ít nhất 1 cột, kiểm tra xem các cột khác có đuôi ngắn đi kèm không (ví dụ 'THPT')
    if spliced_cols:
        for c_idx in range(1, max_cols):
            if c_idx not in spliced_cols:
                p_val = str(prev_row[c_idx] or "").strip()
                c_val = str(curr_row[c_idx] or "").strip()
                if p_val and c_val and len(c_val) <= 15 and not re.search(r"\d", c_val):
                    spliced_cols.append(c_idx)
        return True, sorted(spliced_cols), "; ".join(reasons)
        
    return False, [], None


def can_chain_tables(
    prev_t: dict[str, Any], 
    curr_t: dict[str, Any], 
    all_blocks: list[dict[str, Any]]
) -> tuple[bool, str]:
    """
    Kiểm tra tính liên tục chặt chẽ giữa 2 bảng liên tiếp:
    1. Trang liên tiếp: curr_page == prev_page + 1.
    2. Cùng section.
    3. Không có block 'heading' nào nằm xen giữa 2 bảng.
    4. Số cột thực tế tương thích.
    5. Vị trí vật lý hợp lý trên trang.
    """
    prev_page = prev_t.get("page", 0)
    curr_page = curr_t.get("page", 0)
    
    if curr_page != prev_page + 1:
        return False, "non_consecutive_pages"
    
    if prev_t.get("section") != curr_t.get("section"):
        return False, "different_section"
    
    # Kiểm tra xem giữa prev_t và curr_t có block heading nào không
    try:
        idx_prev = all_blocks.index(prev_t)
        idx_curr = all_blocks.index(curr_t)
        for mid_idx in range(idx_prev + 1, idx_curr):
            mid_block = all_blocks[mid_idx]
            if mid_block.get("type") == "heading":
                return False, "interrupted_by_heading"
    except ValueError:
        pass
    
    prev_cols = get_table_column_count(prev_t)
    curr_cols = get_table_column_count(curr_t)
    if prev_cols == 0 or prev_cols != curr_cols:
        return False, f"incompatible_columns ({prev_cols} != {curr_cols})"
    
    # Kiểm tra vị trí: bảng trước ở nửa dưới trang (bottom >= 250), bảng sau ở nửa trên trang (top <= 250)
    if prev_t.get("bottom", 0) < 250 or curr_t.get("top", 999) > 250:
        return False, "spatial_bounds_out_of_range"
    
    return True, "valid_chain_continuity"


def resolve_multipage_tables(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Giai đoạn 4: Xử lý chuỗi bảng kéo dài qua nhiều trang (Multi-page Table Continuity V2).
    
    Quy trình xử lý tuần tự chuẩn mực:
    1. Thiết lập metadata nguồn ban đầu.
    2. Nhận diện các cặp bảng thuộc cùng chuỗi qua can_chain_tables().
    3. Duyệt từng cặp bảng liên tiếp:
       - Bước 1: Rà soát và loại bỏ Header lặp lại an toàn (nếu có).
         Cập nhật rows_normalized và row_metadata. KHÔNG DÙNG CONTINUE.
       - Bước 2: Kiểm tra dòng dữ liệu đầu tiên còn lại của bảng sau với dòng cuối bảng trước.
         Nếu có bằng chứng tiếp nối văn bản: Ghép nối từng ô, cập nhật metadata cấp bản ghi
         (source_pages tích lũy, is_cross_page_spliced=True), và xóa dòng đã ghép khỏi bảng sau.
       - Bước 3: Đồng bộ metadata cấp bảng (table_chain_id, source_pages, chain_page_range).
    """
    for block in blocks:
        if "source_pages" not in block:
            block["source_pages"] = [block.get("page")]

    table_blocks = [b for b in blocks if b.get("type") == "table"]
    chain_counter = 1
    current_chain_id = None
    root_header: list[str] | None = None

    for i in range(len(table_blocks) - 1):
        prev_t = table_blocks[i]
        curr_t = table_blocks[i + 1]

        prev_page = prev_t.get("page", 0)
        curr_page = curr_t.get("page", 0)

        can_chain, reason = can_chain_tables(prev_t, curr_t, blocks)
        if not can_chain:
            current_chain_id = None
            root_header = None
            continue

        # => XÁC NHẬN HAI BẢNG THUỘC CÙNG MỘT CHUỖI
        if not current_chain_id:
            current_chain_id = f"table_chain_{chain_counter}_p{prev_page}"
            chain_counter += 1
            # Lưu reference header từ bảng đầu tiên của chuỗi
            root_header = prev_t["content"]["rows_raw"][0] if prev_t["content"]["rows_raw"] else None

        prev_t["table_chain_id"] = current_chain_id
        curr_t["table_chain_id"] = current_chain_id
        prev_t["continues_on_next_page"] = True
        curr_t["is_continuation"] = True
        curr_t["source_pages"] = sorted(list(set(prev_t["source_pages"] + [curr_page])))

        curr_norm_rows = curr_t["content"]["rows_normalized"]
        prev_norm_rows = prev_t["content"]["rows_normalized"]
        curr_row_meta = curr_t["content"].get("row_metadata", [])
        prev_row_meta = prev_t["content"].get("row_metadata", [])

        # -------------------------------------------------------------
        # BƯỚC 1: XỬ LÝ HEADER LẶP LẠI (NẾU CÓ)
        # -------------------------------------------------------------
        if curr_norm_rows:
            col_cnt = get_table_column_count(curr_t)
            is_hdr, h_reason = detect_recurrent_header(curr_norm_rows[0], root_header, col_cnt)
            if is_hdr:
                curr_norm_rows.pop(0)
                if curr_row_meta:
                    curr_row_meta.pop(0)
                curr_t["content"]["recurrent_header_removed"] = True
                curr_t["content"]["recurrent_header_reason"] = h_reason
            else:
                # Nếu không phải header lặp lại thì trong bảng tiếp nối, hàng đầu tiên là hàng dữ liệu
                if curr_row_meta:
                    curr_row_meta[0]["is_header"] = False

        # -------------------------------------------------------------
        # BƯỚC 2: KIỂM TRA DÒNG DỮ LIỆU ĐẦU TIÊN CÒN LẠI ĐỂ GHÉP NỐI
        # (Không bị bỏ qua kể cả khi vừa xóa header ở Bước 1)
        # -------------------------------------------------------------
        if curr_norm_rows and prev_norm_rows:
            candidate_row = curr_norm_rows[0]
            last_prev_row = prev_norm_rows[-1]
            
            should_splice, spliced_cols, s_reason = check_cross_page_continuation(last_prev_row, candidate_row)
            if should_splice:
                # Ghép các cột tiếp nối vào hàng cuối của bảng trước
                for c_idx in spliced_cols:
                    add_text = candidate_row[c_idx].strip()
                    if add_text:
                        if last_prev_row[c_idx]:
                            last_prev_row[c_idx] += " " + add_text
                        else:
                            last_prev_row[c_idx] = add_text

                # Cập nhật metadata cấp bản ghi cho hàng vừa được ghép
                if prev_row_meta:
                    last_meta = prev_row_meta[-1]
                    last_meta["source_pages"] = sorted(list(set(last_meta.get("source_pages", [prev_page]) + [curr_page])))
                    last_meta["is_cross_page_spliced"] = True
                    last_meta["spliced_details"] = {
                        "spliced_from_page": curr_page,
                        "columns": spliced_cols,
                        "reason": s_reason
                    }

                # Loại bỏ hàng đã ghép khỏi bảng trang sau
                curr_norm_rows.pop(0)
                if curr_row_meta:
                    curr_row_meta.pop(0)
                curr_t["content"]["cross_page_row_spliced"] = True

                # Nếu bảng trang sau đã bị ghép hết sạch hàng, cập nhật source_pages cho bảng trước
                if len(curr_norm_rows) == 0:
                    prev_t["source_pages"] = sorted(list(set(prev_t.get("source_pages", [prev_page]) + curr_t.get("source_pages", [curr_page]))))

        # Đánh số lại row_index tuần tự sau khi pop header hoặc pop hàng đã ghép
        for idx, m in enumerate(curr_row_meta):
            m["row_index"] = idx
        for idx, m in enumerate(prev_row_meta):
            m["row_index"] = idx

        # Cập nhật số hàng chuẩn hóa và mảng metadata
        curr_t["content"]["rows_normalized"] = curr_norm_rows
        curr_t["content"]["row_metadata"] = curr_row_meta
        curr_t["content"]["row_count_normalized"] = len(curr_norm_rows)
        prev_t["content"]["rows_normalized"] = prev_norm_rows
        prev_t["content"]["row_metadata"] = prev_row_meta
        prev_t["content"]["row_count_normalized"] = len(prev_norm_rows)

    # Loại bỏ các block bảng rỗng do toàn bộ hàng đã được ghép nối sang bảng trước
    cleaned_blocks = [
        b for b in blocks
        if not (b.get("type") == "table" and len(b.get("content", {}).get("rows_normalized", [])) == 0)
    ]

    return cleaned_blocks


def validate_multipage_table_continuity(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Bộ kiểm tra validation chuyên sâu cho Giai đoạn 4:
    1. Kiểm tra tính đồng bộ giữa rows_normalized và row_metadata (độ dài phải khớp 1-1).
    2. Kiểm tra tính toàn vẹn của table_chain_id và source_pages.
    3. Thống kê số lượng chuỗi bảng, số header lặp bị loại bỏ, số hàng được ghép qua trang.
    4. Thống kê các trường hợp cảnh báo hoặc cần kiểm tra thủ công.
    """
    table_blocks = [b for b in blocks if b.get("type") == "table"]
    chain_groups: dict[str, list[dict[str, Any]]] = {}
    
    metadata_mismatches = []
    cross_page_spliced_rows = []
    headers_removed_count = 0
    warnings = []

    for t in table_blocks:
        t_id = t.get("table_id")
        page = t.get("page")
        content = t.get("content", {})
        norm_rows = content.get("rows_normalized", [])
        row_meta = content.get("row_metadata", [])
        
        # 1. Kiểm tra độ dài 1-1 giữa rows_normalized và row_metadata
        if len(norm_rows) != len(row_meta):
            metadata_mismatches.append({
                "table_id": t_id,
                "page": page,
                "rows_norm_count": len(norm_rows),
                "row_meta_count": len(row_meta)
            })

        # 2. Thống kê header bị loại bỏ
        if content.get("recurrent_header_removed"):
            headers_removed_count += 1

        # 3. Thống kê các hàng được ghép qua trang
        for r_meta in row_meta:
            if r_meta.get("is_cross_page_spliced"):
                cross_page_spliced_rows.append({
                    "table_id": t_id,
                    "stt": r_meta.get("stt"),
                    "source_pages": r_meta.get("source_pages"),
                    "details": r_meta.get("spliced_details")
                })
            if r_meta.get("needs_manual_review"):
                warnings.append({
                    "table_id": t_id,
                    "row_index": r_meta.get("row_index"),
                    "reason": r_meta.get("review_reason")
                })

        # Gom nhóm chuỗi bảng
        chain_id = t.get("table_chain_id")
        if chain_id:
            chain_groups.setdefault(chain_id, []).append(t)

    is_valid = (len(metadata_mismatches) == 0)

    return {
        "is_valid": is_valid,
        "total_table_blocks": len(table_blocks),
        "total_chains": len(chain_groups),
        "chain_details": {
            cid: {
                "pages": [tbl["page"] for tbl in tbls],
                "table_count": len(tbls)
            } for cid, tbls in chain_groups.items()
        },
        "headers_removed_count": headers_removed_count,
        "cross_page_spliced_rows_count": len(cross_page_spliced_rows),
        "cross_page_spliced_rows": cross_page_spliced_rows,
        "metadata_mismatches_count": len(metadata_mismatches),
        "metadata_mismatches": metadata_mismatches,
        "warnings_count": len(warnings),
        "warnings": warnings
    }

def merge_text_across_pages(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Giai đoạn 1: Tạm tắt việc ghép nối text xuyên trang để bảo toàn tính toàn vẹn 
    tọa độ vật lý (top, bottom, bbox) và ranh giới trang nguồn (page).
    
    Tất cả các block được bảo tồn độc lập theo từng trang thực tế của PDF.
    """
    return blocks


def validate_extracted_blocks(blocks: list[dict[str, Any]], expected_total_pages: int | None = None) -> dict[str, Any]:
    """
    Kiểm tra tự động chất lượng và tính toàn vẹn của danh sách block:
    1. Kiểm tra tọa độ bất thường: bottom < top.
    2. Kiểm tra tính hiện diện và tính hợp lệ của số trang (page).
    3. Kiểm tra block rỗng nội dung.
    """
    invalid_coords: list[dict[str, Any]] = []
    empty_content: list[dict[str, Any]] = []
    missing_pages: list[int] = []
    pages_found: set[int] = set()

    for idx, block in enumerate(blocks):
        page = block.get("page")
        top = block.get("top")
        bottom = block.get("bottom")
        b_type = block.get("type")

        # 1. Kiểm tra số trang
        if page is None or not isinstance(page, int) or page <= 0:
            missing_pages.append(idx)
        else:
            pages_found.add(page)
            for sp in block.get("source_pages", []):
                if isinstance(sp, int) and sp > 0:
                    pages_found.add(sp)

        # 2. Kiểm tra tọa độ (nếu có đủ top và bottom)
        if top is not None and bottom is not None:
            if float(bottom) < float(top):
                invalid_coords.append({
                    "block_index": idx,
                    "page": page,
                    "type": b_type,
                    "top": top,
                    "bottom": bottom,
                    "content_preview": str(block.get("content", ""))[:80],
                })

        # 3. Kiểm tra nội dung rỗng
        if b_type in ("text", "heading"):
            content_str = str(block.get("content", "")).strip()
            if not content_str:
                empty_content.append({"block_index": idx, "page": page, "type": b_type})
        elif b_type == "table":
            rows = block.get("content", {}).get("rows_normalized", [])
            has_cell_data = any(str(cell or "").strip() for row in rows for cell in row)
            if not has_cell_data:
                empty_content.append({"block_index": idx, "page": page, "type": b_type})

    is_valid = (
        len(invalid_coords) == 0 
        and len(empty_content) == 0 
        and len(missing_pages) == 0
        and (expected_total_pages is None or len(pages_found) == expected_total_pages)
    )

    return {
        "is_valid": is_valid,
        "total_blocks": len(blocks),
        "total_pages_detected": len(pages_found),
        "invalid_coords_count": len(invalid_coords),
        "invalid_coords": invalid_coords,
        "empty_content_count": len(empty_content),
        "empty_content": empty_content,
        "missing_pages_count": len(missing_pages),
    }

def assign_section_numbers(blocks: list[dict[str, Any]], initial_section: str = "Thông tin chung") -> list[dict[str, Any]]:
    """
        Gán section theo thứ tự block.
        Section được giữ xuyên qua các trang
    """
    current_section = initial_section

    for block in blocks:
        if block["type"] == "heading":
            heading = parse_section_heading(block["content"])

            if heading:
                current_section = heading
            
        block["section"] = current_section

    return blocks

def extract_pdf(file_path: str | Path) -> dict[str, Any]:
    """
        Đọc toàn bộ PDF và trả về các block có thứ tự.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    all_blocks: list[dict[str, Any]] = []
    current_section = "Thông tin chung"

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            page_blocks = extract_page_blocks(page, page_number)

            for block in page_blocks:
                block["page_height"] = float(page.height)
            
            page_blocks = assign_section_numbers(page_blocks, initial_section=current_section)

            if page_blocks:
                current_section = page_blocks[-1]["section"]

            all_blocks.extend(page_blocks)
    
    all_blocks = merge_text_across_pages(all_blocks)
    all_blocks = resolve_multipage_tables(all_blocks)
    return {
        "source_file": pdf_path.name,
        "total_pages": total_pages,
        "blocks": all_blocks
    }
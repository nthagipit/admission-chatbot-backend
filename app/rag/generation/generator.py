import logging
from typing import Any

from app.rag.generation.prompts import CONTEXT_PROMPT
from app.rag.generation.llm_builder import llm_builder, llm_fast_builder

logger = logging.getLogger(__name__)

FALLBACK_NO_INFO_ANSWER = (
    "Thông tin này hiện chưa có trong tài liệu tuyển sinh được cung cấp. "
    "Bạn vui lòng liên hệ phòng tuyển sinh để được giải đáp chi tiết."
)

FALLBACK_ERROR_ANSWER = (
    "Hệ thống đang gặp sự cố tạm thời khi kết nối với mô hình AI. "
    "Bạn vui lòng thử lại sau giây lát hoặc liên hệ phòng tuyển sinh."
)


def _format_page_display(chunk: dict[str, Any]) -> str:
    """Định dạng chuỗi hiển thị trang (hỗ trợ nhiều trang nguồn liên tiếp)."""
    source_pages = chunk.get("source_pages")
    if source_pages and isinstance(source_pages, list):
        valid_pages = [str(p) for p in source_pages if p is not None]
        if valid_pages:
            return ", ".join(valid_pages)
    page = chunk.get("page_number")
    if page is not None:
        return str(page)
    return "Không rõ trang"


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Gộp nội dung các chunk thành chuỗi ngữ cảnh đưa vào prompt.
    Bao gồm tên tài liệu, trang (hỗ trợ đa trang) và mục (section).
    """
    if not chunks:
        return "Không tìm thấy tài liệu phù hợp."

    formatted_chunks = []
    for idx, chunk in enumerate(chunks, 1):
        pages_str = _format_page_display(chunk)
        doc_title = chunk.get("document_title") or "Tài liệu tuyển sinh"
        section = chunk.get("section")
        section_info = f", Mục: {section}" if section else ""
        content = chunk.get("content", "").strip()

        formatted_chunks.append(
            f"--- Đoạn {idx} (Tài liệu: {doc_title}, Trang: {pages_str}{section_info}) ---\n{content}"
        )
    return "\n\n".join(formatted_chunks)


def extract_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Trích xuất danh sách nguồn không trùng lặp (document_title, page_number, source_pages, section).
    """
    citations = []
    seen = set()
    for chunk in chunks:
        doc_title = chunk.get("document_title") or "Tài liệu tuyển sinh"
        page_num = chunk.get("page_number")
        source_pages = chunk.get("source_pages") or ([page_num] if page_num is not None else [])
        section = chunk.get("section")
        key = (doc_title, tuple(source_pages) if source_pages else page_num)
        if key not in seen:
            seen.add(key)
            citations.append({
                "document_title": doc_title,
                "page_number": page_num,
                "source_pages": source_pages,
                "section": section,
            })
    return citations


def _extract_text(content: Any) -> str:
    """
    Trích xuất plain text từ response.content.
    - Nếu là string → trả về trực tiếp.
    - Nếu là list of dicts → ghép các block type='text' (format Gemini content blocks).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block
        ]
        return "\n".join(parts)
    return str(content)


def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Nhận query và retrieved chunks, gọi LLM và trả về answer + citations.
    - Nếu chunks rỗng: trả về thông báo chuẩn ngay lập tức (US03), không tốn quota LLM.
    - Bọc xử lý lỗi API/mạng an toàn, không làm gián đoạn luồng trò chuyện.
    """
    # 1. Fallback sớm khi không có tài liệu phù hợp (US03)
    if not chunks:
        return {
            "answer": FALLBACK_NO_INFO_ANSWER,
            "citations": [],
        }

    # 2. Tạo prompt và gọi LLM
    try:
        context = format_context(chunks)
        
        formatted_prompt = CONTEXT_PROMPT.format(context=context, query=query)

        print("\nContext:", context)

        response = llm_fast_builder.invoke(formatted_prompt)
        raw_answer = _extract_text(response.content).strip()

        answer = raw_answer if raw_answer else FALLBACK_NO_INFO_ANSWER
        return {
            "answer": answer,
            "citations": extract_citations(chunks),
        }
    except Exception as e:
        logger.error("Lỗi khi gọi mô hình sinh câu trả lời LLM: %s", e, exc_info=True)
        return {
            "answer": FALLBACK_ERROR_ANSWER,
            "citations": extract_citations(chunks),
        }

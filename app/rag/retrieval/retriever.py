import logging
import re
from typing import Any
from sqlalchemy import func, or_, case
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.rag.ingestion.embedding import embedder
from app.models.knowledge_document_model import KnowledgeDocument
from app.models.document_chunk_model import DocumentChunk

logger = logging.getLogger(__name__)


def extract_keywords(query: str) -> list[str]:
    """
    Trích xuất các cụm từ khóa có nghĩa từ câu hỏi.
    - Ưu tiên cụm từ dài (4-gram, 3-gram, 2-gram) như tên ngành, chủ đề tuyển sinh.
    - Loại bỏ hư từ và từ chỉ quan hệ để tránh làm loãng tìm kiếm.
    - Loại bỏ cụm con nếu đã có cụm dài hơn bao chứa nó.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return []

    stop_words = {
        "những", "nào", "gì", "như", "thế", "là", "của", "cho",
        "với", "và", "các", "có", "bao", "nhiêu", "bao nhiêu", "ở", "tại", "thì",
        "được", "không", "ạ", "nhỉ", "hỏi", "xin", "về", "ngành", "trường", "khoa"
    }

    clean_query = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
    words = [w.lower() for w in clean_query.split() if w.strip()]

    # 1. Trích xuất n-grams từ 4 từ xuống 2 từ không dính stop words
    candidates = []
    for n in (4, 3, 2):
        for i in range(len(words) - n + 1):
            ngram = words[i : i + n]
            if ngram[0] in stop_words or ngram[-1] in stop_words:
                continue
            if any(w in stop_words for w in ngram):
                continue
            candidates.append(" ".join(ngram))

    # 2. Giữ lại số hoặc mã tuyển sinh nếu có (ví dụ: '7480201', '2026')
    for w in words:
        if re.match(r"^\d+$", w):
            candidates.append(w)

    # 3. Lọc bỏ các cụm từ con bị bao phủ bởi cụm từ dài hơn (tránh lặp từ gây loãng)
    phrases = []
    for c in sorted(candidates, key=len, reverse=True):
        if not any(c in p and c != p for p in phrases):
            phrases.append(c)

    # 4. Fallback: Nếu câu hỏi quá ngắn không có cụm 2 từ, mới lấy từ đơn (>= 3 ký tự)
    if not phrases:
        phrases = [w for w in words if len(w) >= 3 and w not in stop_words]

    return phrases


# ──────────────────────────────────────────────────────────────────────
# Keyword Weighting — topic keywords được ưu tiên hơn entity keywords
# ──────────────────────────────────────────────────────────────────────

# Các cụm từ chủ đề (topic keywords): biểu thị loại thông tin được hỏi
# (học phí, điểm chuẩn, phương thức...) — khác với entity keyword là tên ngành.
_TOPIC_TERMS: frozenset[str] = frozenset({
    "học phí", "điểm chuẩn", "điểm trúng tuyển", "phương thức",
    "phương thức xét tuyển", "phương thức xt", "xét tuyển",
    "học bổng", "chỉ tiêu", "đăng ký", "hồ sơ",
    "thủ tục", "ngưỡng đảm bảo", "tổ hợp", "điều kiện",
    "tuyển thẳng", "miễn giảm", "lệ phí", "tín chỉ",
    "học bổng khuyến khích", "ký túc xá", "hỗ trợ tài chính",
    "mức thu", "học phí dự kiến", "điểm ưu tiên",
    "giải thiểu", "giải thiểu ngành",
})
_TOPIC_WEIGHT = 3    # Trọng số cho topic keyword (học phí, điểm chuẩn...)
_ENTITY_WEIGHT = 1   # Trọng số cho entity keyword (tên ngành, mã ngành...)


def _keyword_weight(kw: str) -> int:
    """
    Trả về trọng số SQL cho từ khóa trong keyword_score.

    - Topic keyword (học phí, điểm chuẩn, ...): weight = _TOPIC_WEIGHT (3)
      → Chunk chứa topic keyword được xếp hạng cao hơn trong keyword search.
    - Entity keyword (tên ngành, mã ngành, ...): weight = _ENTITY_WEIGHT (1)
      → Chunk chỉ nhắc tên ngành mà không chứa thông tin cần hỏi không bị ưu tiên.
    """
    return _TOPIC_WEIGHT if kw.lower() in _TOPIC_TERMS else _ENTITY_WEIGHT


# ──────────────────────────────────────────────────────────────────────
# Topic Detection & Heuristic Reranking
# ──────────────────────────────────────────────────────────────────────

TOPIC_TUITION = "tuition"
TOPIC_BENCHMARK = "benchmark"
TOPIC_ADMISSION_METHOD = "admission_method"
TOPIC_UNKNOWN = "unknown"

_TUITION_KEYWORDS = (
    "học phí", "mức thu", "học bổng", "tín chỉ", "tiền học", "chi phí",
    "bao nhiêu tiền", "đóng bao nhiêu", "lệ phí", "hỗ trợ tài chính",
)
_TUITION_POLICY_KEYWORDS = (
    "miễn giảm", "giảm học phí", "chính sách miễn", "đối tượng miễn",
    "miễn học phí", "chế độ miễn giảm", "trợ cấp",
)
_BENCHMARK_KEYWORDS = (
    "điểm chuẩn", "điểm trúng tuyển", "điểm sàn", "ngưỡng đầu vào",
    "bao nhiêu điểm đỗ", "lấy bao nhiêu điểm", "thang điểm", "điểm xét tuyển",
)
_METHOD_KEYWORDS = (
    "phương thức", "phương thức xét tuyển", "phương thức tuyển sinh",
    "xét tuyển", "xét học bạ", "tuyển thẳng", "đánh giá năng lực",
    "đánh giá tư duy", "tổ hợp", "tổ hợp môn", "điều kiện xét tuyển",
)

_HEADING_NUMBER_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})*|[IVXLCDM]+)\.?\s+([^\n]+)$",
    re.IGNORECASE,
)


def detect_query_topic(query: str) -> dict[str, Any]:
    """
    Xác định chủ đề chính của câu hỏi và trích xuất entities liên quan.
    Hỗ trợ tối thiểu các nhóm: học phí (mức thu & miễn giảm), điểm chuẩn, phương thức xét tuyển.
    """
    q_lower = query.lower()
    is_waiver = any(kw in q_lower for kw in _TUITION_POLICY_KEYWORDS)
    is_tuition = any(kw in q_lower for kw in _TUITION_KEYWORDS) or is_waiver
    is_benchmark = any(kw in q_lower for kw in _BENCHMARK_KEYWORDS)
    is_method = any(kw in q_lower for kw in _METHOD_KEYWORDS)

    if is_waiver:
        topic = TOPIC_TUITION
        sub_topic = "waiver_policy"
    elif is_tuition:
        topic = TOPIC_TUITION
        sub_topic = "fee"
    elif is_benchmark:
        topic = TOPIC_BENCHMARK
        sub_topic = "cutoff_score"
    elif is_method:
        topic = TOPIC_ADMISSION_METHOD
        sub_topic = "methods"
    else:
        topic = TOPIC_UNKNOWN
        sub_topic = "general"

    extracted = extract_keywords(query)
    entities = [
        kw for kw in extracted
        if kw.lower() not in _TOPIC_TERMS and not any(kw.lower() in t for t in _TOPIC_TERMS)
    ]

    return {
        "topic": topic,
        "sub_topic": sub_topic,
        "is_waiver": is_waiver,
        "entities": entities,
    }


def is_header_only_chunk(content: str, metadata: dict[str, Any] | None = None) -> bool:
    """
    Nhận diện chunk chỉ là tiêu đề đề mục, không chứa nội dung trả lời câu hỏi.
    Áp dụng heuristic dựa trên cả metadata lẫn cấu trúc văn bản.
    """
    if metadata and metadata.get("content_type") == "heading":
        return True

    text = content.strip()
    # Bảng markdown không phải header-only
    if "|" in text:
        return False

    # Nếu dài hơn 150 ký tự thì là đoạn văn bản mô tả, không phải header-only trơ trọi
    if len(text) > 150:
        return False

    # Loại bỏ tiền tố context "Phan: ..." nếu có
    clean_lines = [
        l.strip() for l in text.split("\n")
        if l.strip() and not l.strip().startswith("Phan:")
    ]
    if not clean_lines:
        return True

    clean_body = " ".join(clean_lines)
    match = _HEADING_NUMBER_RE.match(clean_body)
    if len(clean_body) <= 120 and match:
        _h_num, h_title = match.groups()
        # Kiểm tra xem tiêu đề mục có chứa dữ liệu định lượng cụ thể (tiền tệ hoặc điểm số) hay không
        has_specific_data = bool(
            re.search(
                r"(\d{1,3}[\.,]\d{3}|\b\d{1,2}(?:[,\.]\d{1,2})?\s*(?:triệu|đồng|vnđ|đ|điểm)\b)",
                h_title.lower(),
            )
        )
        if not has_specific_data:
            return True

    return False


def calculate_rerank_adjustment(
    candidate: dict[str, Any],
    topic_info: dict[str, Any],
) -> tuple[float, str]:
    """
    Tính điểm điều chỉnh (adjustment) cho candidate dựa trên:
    1. Heuristic header-only: phạt nặng nếu chunk chỉ là tiêu đề trơ trọi.
    2. Direct Answer Signal: thưởng nếu chunk chứa câu trả lời trực tiếp (số tiền, điểm số, phương thức).
    3. Topic Mismatch: phạt nếu chunk thuộc bảng/section của chủ đề khác không có thông tin cần hỏi.
    4. Entity Match: thưởng thêm nếu chunk vừa khớp chủ đề vừa chứa entity (tên ngành/mã ngành) được hỏi.
    """
    content = candidate.get("content", "")
    content_lower = content.lower()
    metadata = candidate.get("chunk_metadata") or {}
    section = str(candidate.get("section") or "").lower()

    # Heuristic 1: Header-only chunk
    if is_header_only_chunk(content, metadata):
        return -0.020, "header_only_penalty: chunk chỉ chứa tiêu đề, không có dữ liệu trả lời"

    topic = topic_info["topic"]
    sub_topic = topic_info["sub_topic"]
    adjustment = 0.0
    reasons = []

    if topic == TOPIC_TUITION:
        has_money_rates = bool(
            re.search(
                r"(đồng/tín chỉ|đ/tín chỉ|triệu đồng|nghìn đồng|vnđ|mức thu|học phí:|567\.100|\d{1,3}\.\d{3}\s*(?:đồng|đ))",
                content_lower,
            )
        )
        is_waiver_content = any(
            kw in content_lower
            for kw in ("miễn giảm", "chính sách miễn", "đối tượng được miễn", "chế độ miễn giảm", "giảm 70%", "giảm 100%")
        )

        if sub_topic == "waiver_policy":
            if is_waiver_content:
                adjustment += 0.015
                reasons.append("topic_direct: chứa chính sách/đối tượng miễn giảm học phí")
            elif has_money_rates:
                adjustment += 0.002
                reasons.append("topic_related: chứa mức thu học phí")
            elif "điểm" in section or "trúng tuyển" in section or "chỉ tiêu" in section:
                adjustment -= 0.010
                reasons.append("topic_mismatch: bảng xét tuyển/điểm chuẩn cho câu hỏi miễn giảm")
        else:  # sub_topic == "fee" (mức thu học phí)
            if has_money_rates:
                adjustment += 0.015
                reasons.append("topic_direct: chứa số tiền/đơn giá tín chỉ/bảng mức thu học phí")
            elif is_waiver_content:
                adjustment -= 0.003
                reasons.append("partial_mismatch: chính sách miễn giảm cho câu hỏi hỏi mức thu học phí")
            elif ("điểm" in section or "trúng tuyển" in section or "chỉ tiêu" in section) and not has_money_rates:
                adjustment -= 0.010
                reasons.append("topic_mismatch: bảng xét tuyển/điểm chuẩn không có thông tin học phí")

    elif topic == TOPIC_BENCHMARK:
        has_cutoff_score = bool(
            re.search(
                r"(điểm chuẩn|điểm trúng tuyển|thang điểm|\b[12]\d\.\d{1,2}\b)",
                content_lower,
            )
        )
        is_cutoff_section = "điểm" in section or "trúng tuyển" in section

        if has_cutoff_score or is_cutoff_section:
            adjustment += 0.015
            reasons.append("topic_direct: chứa điểm chuẩn/điểm trúng tuyển")
        elif "học phí" in section or "567.100" in content_lower:
            adjustment -= 0.010
            reasons.append("topic_mismatch: thông tin học phí cho câu hỏi hỏi điểm chuẩn")

    elif topic == TOPIC_ADMISSION_METHOD:
        has_method_data = any(
            kw in content_lower
            for kw in ("phương thức 1", "phương thức 2", "phương thức 3", "phương thức 4",
                       "phương thức 5", "xét học bạ", "thi tốt nghiệp", "đánh giá năng lực",
                       "tuyển thẳng", "tổ hợp môn", "tổ hợp xét tuyển")
        )
        is_method_section = "phương thức" in section or "xét tuyển" in section

        if has_method_data or is_method_section:
            adjustment += 0.015
            reasons.append("topic_direct: mô tả phương thức xét tuyển/điều kiện tuyển sinh")
        elif "học phí" in section or "567.100" in content_lower:
            adjustment -= 0.010
            reasons.append("topic_mismatch: bảng học phí cho câu hỏi hỏi phương thức xét tuyển")

    # Entity Match bonus (nếu câu hỏi có tên ngành/mã ngành và chunk không bị topic_mismatch)
    if topic_info["entities"] and adjustment >= 0:
        matched_entities = [ent for ent in topic_info["entities"] if ent.lower() in content_lower]
        if matched_entities:
            adjustment += 0.006
            reasons.append(f"entity_match: khớp thực thể {matched_entities}")

    reason_str = "; ".join(reasons) if reasons else "neutral: không có tín hiệu điều chỉnh đặc biệt"
    return adjustment, reason_str


def search_similar_chunks(
    query: str,
    db: Session,
    top_k: int = 4,
    document_id: str | None = None,
) -> list[dict]:
    """
    Hybrid Search = Vector Search + Keyword Search, hợp nhất bằng RRF
    kết hợp Reranking nhẹ nhàng dựa trên chủ đề và mức độ trả lời trực tiếp.

    - Vector search: hiểu ngữ nghĩa câu hỏi (cosine_distance trên pgvector).
    - Keyword search: tìm tên ngành, mã ngành, số liệu cụ thể (ILIKE pattern).
    - RRF (Reciprocal Rank Fusion): kết hợp xếp hạng từ cả hai nguồn với rrf_k=60.
    - Post-RRF Reranking: ưu tiên bằng chứng trực tiếp, loại bỏ nhiễu header-only và lệch chủ đề.
    """
    if not query or not isinstance(query, str) or not query.strip() or top_k < 1:
        return []

    # 1. Embed câu hỏi bằng embed_query (task_type=RETRIEVAL_QUERY)
    query_vector = embedder.embed_query(query)

    # 2. Vector Search — lấy top_k * 3 candidates để RRF có đủ ứng viên
    distance_col = DocumentChunk.embedding.cosine_distance(query_vector)
    vector_query = (
        db.query(
            DocumentChunk,
            KnowledgeDocument.title,
            distance_col.label("distance"),
        )
        .join(KnowledgeDocument, DocumentChunk.document_id == KnowledgeDocument.id)
        .filter(KnowledgeDocument.status == "COMPLETED")
    )
    if document_id:
        vector_query = vector_query.filter(DocumentChunk.document_id == document_id)

    vector_results = (
        vector_query
        .order_by(distance_col.asc())
        .limit(top_k * 3)
        .all()
    )

    # --- Debug: Vector Search results (pre-RRF) ---
    vector_rank_map: dict[str, int] = {}
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "\n%s\nVECTOR SEARCH — %d candidates (pre-RRF)\n%s",
            "─" * 70,
            len(vector_results),
            "─" * 70,
        )
    for _vr, (_vc, _vt, _vd) in enumerate(vector_results, start=1):
        _vcid = str(_vc.id)
        vector_rank_map[_vcid] = _vr
        if logger.isEnabledFor(logging.DEBUG):
            _vmeta = _vc.chunk_metadata or {}
            _vpn = (
                _vmeta.get("page_number")
                if _vmeta.get("page_number") is not None
                else _vmeta.get("page")
            )
            _vsp = _vmeta.get("source_pages", [_vpn] if _vpn is not None else [])
            _vdist = float(_vd)
            logger.debug(
                "[V%02d] chunk=%.8s | dist=%.4f | sim=%.4f | page=%s | src=%s | sec=%s\n"
                "      %.500s",
                _vr, _vcid, _vdist, round(1 - _vdist, 4),
                _vpn, _vsp, _vmeta.get("section"),
                _vc.content.replace("\n", " ↵ "),
            )

    # 3. Keyword Search — dùng wildcard pattern linh hoạt (%từ1%từ2%)
    keywords = extract_keywords(query)
    if keywords:
        patterns = [f"%{'%'.join(kw.split())}%" for kw in keywords]
        keyword_filters = [
            DocumentChunk.content.ilike(pat)
            for pat in patterns
        ]
        kw_weights = [_keyword_weight(kw) for kw in keywords]
        cases = [
            case((condition, weight), else_=0)
            for condition, weight in zip(keyword_filters, kw_weights)
        ]
        keyword_score = cases[0]
        for c in cases[1:]:
            keyword_score = keyword_score + c

        keyword_query = (
            db.query(DocumentChunk, KnowledgeDocument.title)
            .join(KnowledgeDocument, DocumentChunk.document_id == KnowledgeDocument.id)
            .filter(KnowledgeDocument.status == "COMPLETED")
            .filter(or_(*keyword_filters))
        )
        if document_id:
            keyword_query = keyword_query.filter(DocumentChunk.document_id == document_id)

        keyword_results = (
            keyword_query
            .order_by(keyword_score.desc())
            .limit(top_k * 3)
            .all()
        )
        keyword_rank_map: dict[str, int] = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "\n%s\nKEYWORD SEARCH — %d candidates (pre-RRF) | keywords=%s\n%s",
                "─" * 70,
                len(keyword_results),
                keywords,
                "─" * 70,
            )
        for _kr, (_kc, _kt) in enumerate(keyword_results, start=1):
            _kcid = str(_kc.id)
            keyword_rank_map[_kcid] = _kr
            if logger.isEnabledFor(logging.DEBUG):
                _kmeta = _kc.chunk_metadata or {}
                _kpn = (
                    _kmeta.get("page_number")
                    if _kmeta.get("page_number") is not None
                    else _kmeta.get("page")
                )
                _ksp = _kmeta.get("source_pages", [_kpn] if _kpn is not None else [])
                _kcontent_lower = _kc.content.lower()
                _matched = [kw for kw in keywords if kw.lower() in _kcontent_lower]
                _weights_info = {
                    kw: _keyword_weight(kw) for kw in _matched
                }
                logger.debug(
                    "[K%02d] chunk=%.8s | matched=%s weights=%s | page=%s | src=%s | sec=%s\n"
                    "      %.500s",
                    _kr, _kcid, _matched, _weights_info,
                    _kpn, _ksp, _kmeta.get("section"),
                    _kc.content.replace("\n", " ↵ "),
                )
    else:
        keyword_results = []
        keyword_rank_map = {}

    # 4. Gom candidates, tránh trùng lặp
    candidates: dict[str, dict] = {}

    def add_candidate(chunk, doc_title, distance=None) -> str:
        chunk_id = str(chunk.id)
        if chunk_id not in candidates:
            metadata = chunk.chunk_metadata or {}
            page_num = (
                metadata.get("page_number")
                if metadata.get("page_number") is not None
                else metadata.get("page")
            )
            source_pages = metadata.get(
                "source_pages",
                [page_num] if page_num is not None else [],
            )

            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "document_title": doc_title,
                "page_number": page_num,
                "source_pages": source_pages,
                "section": metadata.get("section"),
                "content_type": metadata.get("content_type"),
                "chunk_index": chunk.chunk_index,
                "chunk_metadata": metadata,
                "similarity": round(1 - float(distance), 4) if distance is not None else None,
                "rrf_score": 0.0,
            }
        elif distance is not None and candidates[chunk_id]["similarity"] is None:
            candidates[chunk_id]["similarity"] = round(1 - float(distance), 4)

        return chunk_id

    # 5. Tính RRF score: 1/(k + rank), k=60 là giá trị chuẩn của RRF
    rrf_k = 60

    for rank, (chunk, doc_title, distance) in enumerate(vector_results, start=1):
        chunk_id = add_candidate(chunk, doc_title, distance)
        candidates[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, (chunk, doc_title) in enumerate(keyword_results, start=1):
        chunk_id = add_candidate(chunk, doc_title)
        candidates[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    # 6. Thứ hạng ban đầu theo RRF score
    initial_ranked = sorted(candidates.values(), key=lambda c: c["rrf_score"], reverse=True)
    for initial_rank, cand in enumerate(initial_ranked, start=1):
        cand["rrf_rank"] = initial_rank

    # 7. Nhận diện chủ đề câu hỏi & Reranking nhẹ nhàng sau RRF
    topic_info = detect_query_topic(query)
    for cand in candidates.values():
        adj, reason = calculate_rerank_adjustment(cand, topic_info)
        cand["rerank_adjustment"] = adj
        cand["rerank_reason"] = reason
        cand["rerank_score"] = round(cand["rrf_score"] + adj, 5)

    # 8. Sắp xếp kết quả cuối theo rerank_score
    reranked = sorted(candidates.values(), key=lambda c: c["rerank_score"], reverse=True)

    # --- Debug: Báo cáo chi tiết trước & sau Reranking (Bước C) ---
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "\n%s\nHYBRID RETRIEVAL & RERANKING REPORT — query: %s\n"
            "Topic Detected: %s (sub_topic: %s, entities: %s)\n%s",
            "═" * 78,
            query[:80],
            topic_info["topic"],
            topic_info["sub_topic"],
            topic_info["entities"],
            "─" * 78,
        )
        logger.debug(
            "%-4s | %-4s | %-8s | %-7s | %-7s | %-7s | %-12s | %s",
            "RRF#", "New#", "Chunk", "RRF", "Rerank", "Delta", "Type/Page", "Reason & Section",
        )
        logger.debug("─" * 78)
        for new_rank, rc in enumerate(reranked[:max(top_k * 2, 6)], start=1):
            rcid = rc["chunk_id"]
            delta = rc["rerank_adjustment"]
            delta_str = f"{delta:+.3f}" if delta != 0 else " 0.000"
            ptype = f"{rc.get('content_type') or 'text'}:p{rc.get('page_number')}"
            logger.debug(
                "[%02d] | [#%02d] | %.8s | %.5f | %.5f | %-7s | %-12s | %s (sec: %s)",
                rc["rrf_rank"], new_rank, rcid, rc["rrf_score"], rc["rerank_score"],
                delta_str, ptype, rc["rerank_reason"], rc.get("section"),
            )
            logger.debug("       Preview: %.160s", rc["content"].replace("\n", " ↵ "))
        logger.debug("═" * 78)

    return reranked[:top_k]


if __name__ == "__main__":
    # Bật DEBUG để xem toàn bộ log hybrid retrieval
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
    )
    db = next(get_db())
    try:
        results = search_similar_chunks(
            query="Học phí ngành Công nghệ thông tin là bao nhiêu?",
            db=db,
            top_k=6,
        )
        for result in results:
            print("Document:", result["document_title"])
            print("Page:", result["page_number"])
            print("Similarity:", result["similarity"])
            print("RRF score:", round(result["rrf_score"], 5))
            print("Content:", result["content"])
            print("-" * 50)
    finally:
        db.close()

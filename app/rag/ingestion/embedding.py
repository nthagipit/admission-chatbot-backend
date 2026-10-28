import logging
import math
import random
import re
import time
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


def _parse_retry_delay(error_message: str) -> float | None:
    """
    Phân tích thời gian chờ do Gemini API gợi ý trong error message.

    API có thể chứa các củ pháp khác nhau:
      - 'Please retry in 28.9s'              (message field)
      - "'retryDelay': '28s'"                (details JSON in string repr)
      - "'retryDelay': '28.912s'"            (với số thập phân)

    Trả về số giây (float) nếu tìm thấy, None nếu không.
    """
    # Củ pháp 1: 'Please retry in 28.912s'
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", error_message, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Củ pháp 2: 'retryDelay': '28s' hoặc 'retryDelay': '28.9s'
    m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", error_message)
    if m:
        return float(m.group(1))

    return None


class EmbeddingService:
    MODEL = "models/gemini-embedding-001"
    DIMENSIONS = 768

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.GEMINI_API_KEY
        common_config = {
            "model": self.MODEL,
            "output_dimensionality": self.DIMENSIONS,
            "google_api_key": key,
        }

        # Dùng khi tạo embedding cho chunks trong kho tri thức.
        self.document_embeddings = GoogleGenerativeAIEmbeddings(
            **common_config,
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Dùng khi tạo embedding cho câu hỏi của người dùng.
        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            **common_config,
            task_type="RETRIEVAL_QUERY",
        )

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))

        if not math.isfinite(norm):
            raise ValueError("Embedding chứa giá trị không hữu hạn.")

        if norm == 0:
            raise ValueError("Embedding có vector độ dài bằng 0.")

        return [value / norm for value in vector]

    def embed_document(self, text: str) -> list[float]:
        """Embedding một chunk tài liệu."""
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Không thể embedding văn bản rỗng.")

        vector = self.document_embeddings.embed_documents([text])[0]
        if len(vector) != self.DIMENSIONS:
            raise ValueError(f"Embedding phải có {self.DIMENSIONS} chiều.")
        return self._normalize(vector)

    def embed_query(self, query: str) -> list[float]:
        """Embedding câu hỏi để tìm kiếm trong vector database."""
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Câu hỏi không được để trống.")

        vector = self.query_embeddings.embed_query(query)
        if len(vector) != self.DIMENSIONS:
            raise ValueError(f"Embedding phải có {self.DIMENSIONS} chiều.")
        return self._normalize(vector)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 10,
        max_retries: int = 5,
        inter_batch_delay: float = 8.0,
    ) -> list[list[float]]:
        """Embedding nhiều chunks theo batch, có retry khi gặp lỗi tạm thời.

        Args:
            texts: Danh sách nội dung chunk cần embed.
            batch_size: Số chunk mỗi lần gọi API. Free Tier đếm mỗi chunk là 1 request.
            max_retries: Số lần thử lại tối đa mỗi batch khi gặp lỗi 429/503.
            inter_batch_delay: Giây chờ giữa các batch thành công liên tiếp.
                Free Tier giới hạn 100 RPM; batch_size=10 → cần ≥6s/batch.
                Mặc định 8s để có biên an toàn. Đặt 0.0 khi dùng Paid Tier.
        """
        if batch_size < 1:
            raise ValueError("batch_size phải lớn hơn 0.")

        if max_retries < 1:
            raise ValueError("max_retries phải lớn hơn 0.")

        if not texts:
            return []

        if any(not text or not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Danh sách có chunk rỗng.")

        all_vectors = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                "Embedding batch %s/%s, số chunks: %s",
                batch_num,
                total_batches,
                len(batch),
            )

            for attempt in range(max_retries):
                try:
                    vectors = self.document_embeddings.embed_documents(batch)

                    if len(vectors) != len(batch):
                        raise ValueError(
                            "Số vector trả về không khớp số chunks đầu vào."
                        )

                    if any(len(v) != self.DIMENSIONS for v in vectors):
                        raise ValueError(
                            f"Embedding phải có {self.DIMENSIONS} chiều."
                        )

                    all_vectors.extend(
                        self._normalize(vector) for vector in vectors
                    )
                    # Thêm delay giữa các batch để tránh vượt RPM của Free Tier.
                    # Chỉ sleep nếu còn batch tiếp theo.
                    if inter_batch_delay > 0 and i + batch_size < len(texts):
                        logger.debug(
                            "Inter-batch delay %.1fs trước batch %s/%s",
                            inter_batch_delay,
                            batch_num + 1,
                            total_batches,
                        )
                        time.sleep(inter_batch_delay)
                    break

                except Exception as exc:
                    message = str(exc).upper()
                    retryable = any(
                        marker in message
                        for marker in (
                            "429",
                            "RESOURCE_EXHAUSTED",
                            "503",
                            "UNAVAILABLE",
                            "500",
                            "INTERNAL",
                        )
                    )

                    if not retryable or attempt == max_retries - 1:
                        raise

                    # Đọc retryDelay do API gợi ý (ví dụ: 'Please retry in 28.9s')
                    # Nếu có, dùng đúng giá trị đó; nếu không, dùng exponential backoff.
                    suggested = _parse_retry_delay(str(exc))
                    if suggested is not None:
                        delay = suggested + random.uniform(1, 3)  # thêm buffer nhỏ
                    else:
                        delay = min(15 * (2 ** attempt), 120) + random.uniform(0, 3)

                    logger.warning(
                        "Batch %s thất bại (%s/%s): %s. Thử lại sau %.1fs",
                        batch_num,
                        attempt + 1,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        if len(all_vectors) != len(texts):
            raise RuntimeError("Số embedding cuối cùng không khớp đầu vào.")

        return all_vectors


embedder = EmbeddingService()

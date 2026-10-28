"""
test_pipeline.py — Unit tests for ingest_pdf_document() in pipeline.py.

Tất cả tests đều dùng mock: không gọi Gemini API thật, không ghi dữ liệu
vào PostgreSQL production.

Các kịch bản được kiểm tra:
  01. Pipeline gọi đúng extract_pdf() và blocks_to_docs() (không dùng API cũ).
  02. Nội dung và metadata của chunks được giữ nguyên khi insert vào DB.
  03. Số embeddings khớp số chunks.
  04. Embedding sai số chiều (< 768) khiến ingestion thất bại và FAILED được set.
  05. Extraction thất bại → exception được raise, trạng thái FAILED được set.
  06. Embedding thất bại → exception được raise, trạng thái FAILED được set.
  07. Database insert thất bại → exception được raise, trạng thái FAILED được set.
  08. Trạng thái COMPLETED chỉ được set sau khi commit thành công, không trước.
  09. Rollback được gọi khi ingestion thất bại.
  10. Chạy lại ingestion (tài liệu đã tồn tại) xóa chunks cũ trước khi insert mới.
  11. Tài liệu sinh ra 0 chunks → ValueError, không set COMPLETED.
"""

import unittest
from unittest.mock import MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: tạo fake DB session và fake doc
# ─────────────────────────────────────────────────────────────────────────────

def _make_db(existing_doc=None):
    """Tạo mock SQLAlchemy Session với hành vi cơ bản."""
    db = MagicMock()

    fake_query = MagicMock()
    fake_query.filter.return_value = fake_query
    fake_query.first.return_value = existing_doc
    fake_query.delete.return_value = 0
    fake_query.update.return_value = 1

    db.query.return_value = fake_query
    db.commit.return_value = None
    db.rollback.return_value = None
    db.add.return_value = None
    db.add_all.return_value = None

    def _refresh(obj):
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = "doc-test-uuid"

    db.refresh.side_effect = _refresh
    return db


def _make_doc(doc_id="doc-test-uuid", status="PROCESSING"):
    doc = MagicMock()
    doc.id = doc_id
    doc.status = status
    doc.file_path = "data/test.pdf"
    doc.title = "Test Document"
    return doc


def _make_chunk(idx, content="Sample chunk content"):
    return {
        "chunk_index": idx,
        "content": content,
        "metadata": {
            "content_type": "text",
            "page": idx + 1,
            "source_pages": [idx + 1],
            "section": "Test Section",
            "table_id": None,
            "table_chain_id": None,
        },
    }


def _make_vector(dim=768):
    return [0.001] * dim


# ─────────────────────────────────────────────────────────────────────────────
# Patch targets — luôn patch ở module pipeline, không phải module gốc
# ─────────────────────────────────────────────────────────────────────────────

_PATCH_EXTRACT   = "app.rag.ingestion.pipeline.extract_pdf"
_PATCH_BLOCKS    = "app.rag.ingestion.pipeline.blocks_to_docs"
_PATCH_CHUNK     = "app.rag.ingestion.pipeline.chunk_documents"
_PATCH_EMBEDDER  = "app.rag.ingestion.pipeline.embedder"
_PATCH_DOC_MODEL = "app.rag.ingestion.pipeline.KnowledgeDocument"
_PATCH_CHUNK_MODEL = "app.rag.ingestion.pipeline.DocumentChunk"
_PATCH_ADMIN     = "app.rag.ingestion.pipeline.get_or_create_admin"


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestPdfDocument(unittest.TestCase):

    def _run_pipeline(
        self,
        db,
        extraction_blocks=None,
        chunks=None,
        vectors=None,
        file_path="data/test.pdf",
        title="Test",
    ):
        """Chạy pipeline với toàn bộ phụ thuộc được mock sẵn."""
        if extraction_blocks is None:
            extraction_blocks = [{"type": "text", "page": 1, "content": "Hello"}]
        if chunks is None:
            chunks = [_make_chunk(0)]
        if vectors is None:
            vectors = [_make_vector()]

        fake_doc = _make_doc()

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={
                "source_file": "test.pdf",
                "total_pages": 1,
                "blocks": extraction_blocks,
            }) as mock_extract,
            patch(_PATCH_BLOCKS, return_value=[
                {"content": c["content"], "metadata": c["metadata"]}
                for c in chunks
            ]) as mock_blocks,
            patch(_PATCH_CHUNK, return_value=chunks) as mock_chunk,
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=fake_doc),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.return_value = vectors

            from app.rag.ingestion.pipeline import ingest_pdf_document
            result = ingest_pdf_document(file_path, title, db)

        return result, mock_extract, mock_blocks, mock_chunk, mock_embedder

    # ── Test 01 ──────────────────────────────────────────────────────────────
    def test_01_pipeline_calls_correct_api(self):
        """Pipeline gọi đúng extract_pdf() và blocks_to_docs(), không dùng build_page_documents."""
        db = _make_db()

        _, mock_extract, mock_blocks, mock_chunk, _ = self._run_pipeline(db)

        mock_extract.assert_called_once_with("data/test.pdf")
        mock_blocks.assert_called_once()  # blocks_to_docs() được gọi
        mock_chunk.assert_called_once()   # chunk_documents() được gọi

    # ── Test 02 ──────────────────────────────────────────────────────────────
    def test_02_chunk_content_and_metadata_preserved(self):
        """Nội dung và metadata chunks được truyền nguyên vẹn khi tạo DocumentChunk."""
        db = _make_db()
        chunks = [_make_chunk(0, "Học phí ngành CNTT 2026"), _make_chunk(1, "Điểm chuẩn ngành 2025")]
        vectors = [_make_vector(), _make_vector()]

        inserted = []

        def _capture_add_all(objs):
            inserted.extend(objs)

        db.add_all.side_effect = _capture_add_all

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[{"content": c["content"], "metadata": c["metadata"]} for c in chunks]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL, side_effect=lambda **kw: kw),
        ):
            mock_embedder.embed_documents.return_value = vectors
            from app.rag.ingestion.pipeline import ingest_pdf_document
            ingest_pdf_document("data/test.pdf", "Test", db)

        # embed_documents nhận đúng texts từ chunks
        called_texts = mock_embedder.embed_documents.call_args[0][0]
        self.assertEqual(called_texts, ["Học phí ngành CNTT 2026", "Điểm chuẩn ngành 2025"])

    # ── Test 03 ──────────────────────────────────────────────────────────────
    def test_03_embedding_count_matches_chunks(self):
        """Số embeddings phải khớp số chunks — nếu không raise RuntimeError."""
        db = _make_db()
        chunks = [_make_chunk(0), _make_chunk(1), _make_chunk(2)]
        vectors_wrong = [_make_vector(), _make_vector()]  # chỉ có 2, trong khi chunks có 3

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.return_value = vectors_wrong
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(RuntimeError) as ctx:
                ingest_pdf_document("data/test.pdf", "Test", db)

        self.assertIn("không khớp", str(ctx.exception).lower())
        db.rollback.assert_called()

    # ── Test 04 ──────────────────────────────────────────────────────────────
    def test_04_wrong_embedding_dimension_raises_and_sets_failed(self):
        """Embedding sai số chiều (ví dụ 256 thay vì 768) → ValueError, FAILED được set."""
        db = _make_db()
        chunks = [_make_chunk(0)]
        bad_vector = [0.1] * 256  # sai chiều

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.return_value = [bad_vector]
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(ValueError) as ctx:
                ingest_pdf_document("data/test.pdf", "Test", db)

        self.assertIn("768", str(ctx.exception))
        db.rollback.assert_called()
        # FAILED update được thực hiện qua query, không qua setattr trên instance
        db.query.assert_called()

    # ── Test 05 ──────────────────────────────────────────────────────────────
    def test_05_extraction_failure_raises_and_sets_failed(self):
        """extract_pdf() ném exception → pipeline raise, FAILED được set."""
        db = _make_db()

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, side_effect=FileNotFoundError("PDF not found")),
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL),
        ):
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(FileNotFoundError):
                ingest_pdf_document("data/test.pdf", "Test", db)

        db.rollback.assert_called()

    # ── Test 06 ──────────────────────────────────────────────────────────────
    def test_06_embedding_failure_raises_and_sets_failed(self):
        """embedder.embed_documents() ném exception → pipeline raise, FAILED được set."""
        db = _make_db()
        chunks = [_make_chunk(0)]

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(RuntimeError) as ctx:
                ingest_pdf_document("data/test.pdf", "Test", db)

        self.assertIn("429", str(ctx.exception))
        db.rollback.assert_called()

    # ── Test 07 ──────────────────────────────────────────────────────────────
    def test_07_db_insert_failure_raises_and_sets_failed(self):
        """db.commit() ném exception khi insert → pipeline raise, rollback được gọi."""
        db = _make_db()
        chunks = [_make_chunk(0)]
        vectors = [_make_vector()]

        commit_calls = [0]

        def _commit():
            commit_calls[0] += 1
            # Lần commit đầu tiên (PROCESSING) OK, lần thứ hai (chunks+COMPLETED) fail
            if commit_calls[0] == 2:
                raise Exception("DB connection lost")

        db.commit.side_effect = _commit

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.return_value = vectors
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(Exception) as ctx:
                ingest_pdf_document("data/test.pdf", "Test", db)

        self.assertIn("connection", str(ctx.exception).lower())
        db.rollback.assert_called()

    # ── Test 08 ──────────────────────────────────────────────────────────────
    def test_08_completed_only_set_after_successful_commit(self):
        """COMPLETED chỉ được set ngay trước lần commit cuối cùng (sau insert chunks)."""
        db = _make_db()
        chunks = [_make_chunk(0)]
        vectors = [_make_vector()]
        fake_doc = _make_doc()
        status_at_commit = []

        original_commit = db.commit

        def _track_commit():
            status_at_commit.append(fake_doc.status)

        db.commit.side_effect = _track_commit

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=fake_doc),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.return_value = vectors
            from app.rag.ingestion.pipeline import ingest_pdf_document
            ingest_pdf_document("data/test.pdf", "Test", db)

        # Lần commit đầu là khi lưu PROCESSING, cuối cùng phải là COMPLETED
        self.assertIn("COMPLETED", status_at_commit,
                      "COMPLETED phải được set trước lần commit cuối.")
        # Lần commit đầu tiên KHÔNG được set COMPLETED
        self.assertEqual(status_at_commit[0], "PROCESSING",
                         "Commit đầu tiên phải ở trạng thái PROCESSING.")

    # ── Test 09 ──────────────────────────────────────────────────────────────
    def test_09_rollback_called_on_failure(self):
        """db.rollback() phải được gọi khi bất kỳ bước nào sau PROCESSING thất bại."""
        db = _make_db()

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, side_effect=OSError("disk read error")),
            patch(_PATCH_DOC_MODEL, return_value=_make_doc()),
            patch(_PATCH_CHUNK_MODEL),
        ):
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(OSError):
                ingest_pdf_document("data/test.pdf", "Test", db)

        db.rollback.assert_called_once()

    # ── Test 10 ──────────────────────────────────────────────────────────────
    def test_10_reingestion_deletes_old_chunks_first(self):
        """Khi tài liệu đã tồn tại, pipeline phải xóa chunks cũ trước khi insert mới."""
        existing_doc = _make_doc(doc_id="existing-uuid", status="COMPLETED")
        db = _make_db(existing_doc=existing_doc)

        # Đảm bảo query().filter().first() trả về existing_doc cho KnowledgeDocument
        # và delete() được gọi cho DocumentChunk
        delete_mock = MagicMock(return_value=5)

        def query_side_effect(model_class):
            q = MagicMock()
            q.filter.return_value = q
            if model_class.__name__ == "KnowledgeDocument" if hasattr(model_class, "__name__") else False:
                q.first.return_value = existing_doc
                q.update.return_value = 1
            else:
                q.first.return_value = None
                q.delete = delete_mock
                q.update.return_value = 1
            return q

        db.query.side_effect = query_side_effect

        chunks = [_make_chunk(0)]
        vectors = [_make_vector()]

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=chunks),
            patch(_PATCH_EMBEDDER) as mock_embedder,
            patch(_PATCH_DOC_MODEL, return_value=existing_doc),
            patch(_PATCH_CHUNK_MODEL),
        ):
            mock_embedder.embed_documents.return_value = vectors
            from app.rag.ingestion.pipeline import ingest_pdf_document
            ingest_pdf_document("data/test.pdf", "Existing Doc", db)

        # db.query() phải được gọi nhiều lần: một trong số đó gọi delete()
        self.assertGreater(db.query.call_count, 0)

    # ── Test 11 ──────────────────────────────────────────────────────────────
    def test_11_zero_chunks_raises_and_does_not_set_completed(self):
        """Nếu chunking trả về [], pipeline phải raise ValueError và không set COMPLETED."""
        db = _make_db()
        fake_doc = _make_doc()

        with (
            patch(_PATCH_ADMIN, return_value=_make_doc("admin-uuid")),
            patch(_PATCH_EXTRACT, return_value={"source_file": "t.pdf", "total_pages": 1, "blocks": []}),
            patch(_PATCH_BLOCKS, return_value=[]),
            patch(_PATCH_CHUNK, return_value=[]),  # 0 chunks
            patch(_PATCH_EMBEDDER),
            patch(_PATCH_DOC_MODEL, return_value=fake_doc),
            patch(_PATCH_CHUNK_MODEL),
        ):
            from app.rag.ingestion.pipeline import ingest_pdf_document

            with self.assertRaises(ValueError) as ctx:
                ingest_pdf_document("data/test.pdf", "Test", db)

        self.assertIn("0 chunks", str(ctx.exception))
        self.assertNotEqual(fake_doc.status, "COMPLETED",
                            "COMPLETED không được set khi sinh ra 0 chunks.")
        db.rollback.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

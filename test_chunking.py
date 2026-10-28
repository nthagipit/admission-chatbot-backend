import unittest, copy, json
from pathlib import Path
from app.rag.ingestion.chunking import (
    split_markdown_table, chunk_documents, blocks_to_docs, _rows_to_markdown,
    _extract_chain_headers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table_doc(rows, row_metadata=None, section="Test", page=1,
                    table_id="t1", table_chain_id=None):
    if row_metadata is None:
        row_metadata = [{"is_header": i == 0, "row_index": i} for i in range(len(rows))]
    md = _rows_to_markdown(rows, row_metadata)
    prefix = f"Phan: {section} | Trang: {page}"
    content = f"{prefix}\n\n{md}"
    metadata = {
        "content_type": "table", "page": page, "source_pages": [page],
        "section": section, "table_id": table_id, "table_chain_id": table_chain_id,
    }
    return {"content": content, "metadata": metadata}


def _build_md(n_data, prefix="", row_len=60, n_headers=1):
    lines = []
    if prefix:
        lines += [prefix, ""]
    for h in range(n_headers):
        lines.append("| " + " | ".join([f"H{h+1}C{c}" for c in range(4)]) + " |")
    lines.append("| --- | --- | --- | --- |")
    for i in range(n_data):
        cell = f"D_{i:04d}" + "x" * max(0, row_len - 7)
        lines.append(f"| {cell} | v2 | v3 | v4 |")
    return "\n".join(lines)


def _data_lines_from_chunks(chunks, n_header_rows=1):
    """Collect data rows (skip prefix, headers, separator) from all chunks."""
    result = []
    for ch in chunks:
        in_table = False
        tbl = []
        for line in ch.split("\n"):
            if not in_table:
                if line.strip().startswith("|"):
                    in_table = True
                    tbl.append(line)
            else:
                tbl.append(line)
        skip = n_header_rows + 1  # header rows + separator
        result.extend(tbl[skip:])
    return result


# ---------------------------------------------------------------------------
# Group A: split_markdown_table
# ---------------------------------------------------------------------------

class TestSplitMarkdownTable(unittest.TestCase):

    def test_short_table_single_chunk(self):
        """Test 2: short table fits in one chunk unchanged."""
        content = _build_md(2, row_len=10)
        result = split_markdown_table(content, max_chunk_chars=1200)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], content)

    def test_long_table_splits(self):
        """Test 3: long table is split into multiple chunks."""
        content = _build_md(20, row_len=60)
        result = split_markdown_table(content, max_chunk_chars=400)
        self.assertGreater(len(result), 1)

    def test_single_row_header_repeated(self):
        """Test 4: single-row header repeated in every chunk."""
        content = _build_md(20, row_len=60)
        result = split_markdown_table(content, max_chunk_chars=400)
        hdr = "| H1C0 | H1C1 | H1C2 | H1C3 |"
        for i, ch in enumerate(result):
            self.assertIn(hdr, ch, f"Chunk {i} missing header")

    def test_multi_row_header_repeated(self):
        """Test 5: multi-row header: both header rows repeated in every chunk."""
        content = _build_md(20, row_len=60, n_headers=2)
        result = split_markdown_table(content, max_chunk_chars=500)
        self.assertGreater(len(result), 1)
        for i, ch in enumerate(result):
            self.assertIn("| H1C0 |", ch, f"Chunk {i} missing header row 1")
            self.assertIn("| H2C0 |", ch, f"Chunk {i} missing header row 2")

    def test_context_prefix_preserved(self):
        """Test 6: context prefix is repeated in every child chunk."""
        prefix = "Phan: Tuyen sinh | Trang: 31, 32"
        content = _build_md(20, prefix=prefix, row_len=60)
        result = split_markdown_table(content, max_chunk_chars=400)
        self.assertGreater(len(result), 1)
        for i, ch in enumerate(result):
            self.assertIn(prefix, ch, f"Chunk {i} missing prefix")

    def test_empty_content(self):
        """Test 7a: empty string returns empty list."""
        self.assertEqual(split_markdown_table(""), [])
        self.assertEqual(split_markdown_table("   "), ["   "])

    def test_malformed_no_separator_preserved(self):
        """Test 7b: malformed table (no separator) returned as-is."""
        content = "| A | B |\n| d1 | d2 |"
        result = split_markdown_table(content, max_chunk_chars=10)
        self.assertEqual(result, [content])

    def test_plain_text_preserved(self):
        """Test 7c: non-table text returned unchanged."""
        content = "Just plain text"
        result = split_markdown_table(content)
        self.assertEqual(result, [content])

    def test_oversized_row_not_truncated(self):
        """Test 8: a single row exceeding the limit is kept intact."""
        big = "X" * 1100
        content = f"| Header |\n| --- |\n| {big} |"
        result = split_markdown_table(content, max_chunk_chars=200)
        self.assertEqual(len(result), 1)
        self.assertIn(big, result[0])

    def test_row_order_preserved(self):
        """Test 9: data row order is exactly maintained across chunks."""
        n = 15
        lines = ["| ID | Data |", "| --- | --- |"]
        for i in range(n):
            lines.append(f"| {i:03d} | ROW_{i:03d} |")
        content = "\n".join(lines)
        result = split_markdown_table(content, max_chunk_chars=200)
        collected = []
        for ch in result:
            for line in ch.split("\n"):
                s = line.strip()
                if s.startswith("|") and "ID" not in s and "---" not in s:
                    collected.append(s)
        expected = [f"| {i:03d} | ROW_{i:03d} |" for i in range(n)]
        self.assertEqual(collected, expected)

    def test_no_empty_chunks(self):
        """Test 11a: no chunk in the output is empty."""
        content = _build_md(10, row_len=60)
        result = split_markdown_table(content, max_chunk_chars=300)
        for ch in result:
            self.assertTrue(ch.strip(), "Got empty chunk")

    def test_no_row_duplication(self):
        """Test 11b: total data rows across chunks equals original count."""
        n = 10
        content = _build_md(n, row_len=60)
        result = split_markdown_table(content, max_chunk_chars=300)
        data_lines = _data_lines_from_chunks(result, n_header_rows=1)
        self.assertEqual(len(data_lines), n)


# ---------------------------------------------------------------------------
# Group B: chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments(unittest.TestCase):

    def test_text_splitting(self):
        """Test 1: normal text is split by RecursiveCharacterTextSplitter."""
        long_text = ("Thong tin tuyen sinh nam 2026. " * 60).strip()
        doc = {
            "content": long_text,
            "metadata": {"content_type": "text", "page": 1, "source_pages": [1],
                         "section": "Intro", "table_id": None, "table_chain_id": None},
        }
        chunks = chunk_documents([doc])
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertTrue(c["content"].strip())
            self.assertIn("chunk_index", c)
            self.assertIn("doc_chunk_index", c)

    def test_metadata_not_mutated(self):
        """Test 10: original metadata dict is never mutated."""
        original_meta = {
            "content_type": "table", "page": 5, "source_pages": [5],
            "section": "S", "table_id": "p5_t0", "table_chain_id": None,
        }
        snapshot = copy.deepcopy(original_meta)
        doc = _make_table_doc(rows=[["H1", "H2"], ["d1", "d2"]])
        doc["metadata"] = original_meta
        chunk_documents([doc])
        self.assertEqual(original_meta, snapshot)

    def test_global_index_monotonic_and_gap_free(self):
        """Global chunk_index is 0-based, monotonic, gap-free across docs."""
        docs = [
            _make_table_doc([["H", "H2"], ["r1", "v1"]], page=1),
            _make_table_doc([["H", "H2"], ["r2", "v2"]], page=2),
        ]
        chunks = chunk_documents(docs)
        indices = [c["chunk_index"] for c in chunks]
        self.assertEqual(indices, list(range(len(indices))))

    def test_doc_chunk_index_per_doc(self):
        """doc_chunk_index restarts at 0 for each document."""
        big_rows = [["H1", "H2"]] + [["d" * 100, "v" * 100] for _ in range(15)]
        rmeta = [{"is_header": i == 0, "row_index": i} for i in range(len(big_rows))]
        md = _rows_to_markdown(big_rows, rmeta)
        content = "Phan: S | Trang: 1\n\n" + md
        doc = {
            "content": content,
            "metadata": {"content_type": "table", "page": 1, "source_pages": [1],
                         "section": "S", "table_id": "t", "table_chain_id": None},
        }
        chunks = chunk_documents([doc])
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c["doc_chunk_index"] for c in chunks], list(range(len(chunks))))

    def test_no_empty_chunks_from_chunk_documents(self):
        """Test 11: empty-content docs are silently skipped; no empty chunks output."""
        docs = [
            {"content": "", "metadata": {"content_type": "text", "page": 1,
                                         "source_pages": [1], "section": "",
                                         "table_id": None, "table_chain_id": None}},
            _make_table_doc([["H"], ["d"]]),
        ]
        chunks = chunk_documents(docs)
        for c in chunks:
            self.assertTrue(c["content"].strip())


# ---------------------------------------------------------------------------
# Group C: blocks_to_docs adapter
# ---------------------------------------------------------------------------

class TestBlocksToDocs(unittest.TestCase):

    def _make_table_block(self, rows, row_meta, page=31):
        return {
            "type": "table",
            "content": {
                "rows_normalized": rows, "row_metadata": row_meta,
                "rows_raw": rows, "row_count": len(rows),
                "row_count_normalized": len(rows),
                "column_counts": [len(rows[0])], "structure_features": {},
            },
            "section": "Muc 11", "page": page, "source_pages": [page],
            "table_id": f"p{page}_t0", "table_chain_id": "table_chain_8_p31",
            "classification": "table_candidate_kept", "continues_on_next_page": False,
        }

    def test_text_block_conversion(self):
        """Test 12a: text block converts content and metadata correctly."""
        docs = blocks_to_docs([{
            "type": "text", "content": "Hello", "section": "S",
            "page": 1, "source_pages": [1],
        }])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "Hello")
        self.assertEqual(docs[0]["metadata"]["content_type"], "text")
        self.assertIsNone(docs[0]["metadata"]["table_id"])

    def test_table_block_renders_markdown(self):
        """Test 12b: table block is rendered as Markdown with context prefix."""
        rows = [["STT", "Ten", "Ma"], ["1.", "CNTT", "7480201"]]
        rmeta = [
            {"is_header": True, "row_index": 0, "stt": "STT",
             "source_pages": [31], "is_cross_page_spliced": False},
            {"is_header": False, "row_index": 1, "stt": "1.",
             "source_pages": [31], "is_cross_page_spliced": False},
        ]
        docs = blocks_to_docs([self._make_table_block(rows, rmeta)])
        self.assertEqual(len(docs), 1)
        self.assertIn("| STT |", docs[0]["content"])
        self.assertIn("7480201", docs[0]["content"])
        self.assertIn("Trang: 31", docs[0]["content"])
        self.assertEqual(docs[0]["metadata"]["table_chain_id"], "table_chain_8_p31")

    def test_empty_table_block_skipped(self):
        """Test 12c: table block with no rows produces no doc."""
        block = {
            "type": "table",
            "content": {"rows_normalized": [], "row_metadata": [], "rows_raw": [],
                        "row_count": 0, "row_count_normalized": 0,
                        "column_counts": [], "structure_features": {}},
            "section": "S", "page": 1, "source_pages": [1],
            "table_id": "p1_t0", "table_chain_id": None,
        }
        self.assertEqual(blocks_to_docs([block]), [])

    def test_empty_text_block_skipped(self):
        """Test 12d: text block with blank content is silently skipped."""
        docs = blocks_to_docs([{
            "type": "text", "content": "  ", "section": "", "page": 1, "source_pages": [1],
        }])
        self.assertEqual(docs, [])

    def test_12e_heading_prepended_to_text(self):
        """Heading block được gộp vào text block tiếp theo, không tạo doc tiêu đề trơ trọi."""
        blocks = [
            {"type": "heading", "content": "10.1. Thông tin về học phí", "section": "10.1. Thông tin về học phí", "page": 28, "source_pages": [28]},
            {"type": "text", "content": "Mức học phí năm 2026 dự kiến...", "section": "10.1. Thông tin về học phí", "page": 28, "source_pages": [28]},
        ]
        docs = blocks_to_docs(blocks)
        self.assertEqual(len(docs), 1)
        self.assertIn("10.1. Thông tin về học phí", docs[0]["content"])
        self.assertIn("Mức học phí năm 2026 dự kiến...", docs[0]["content"])
        self.assertEqual(docs[0]["metadata"]["content_type"], "text")

    def test_12f_heading_absorbed_into_table(self):
        """Heading block đứng trước table không bị tách thành chunk riêng độc lập."""
        rows = [["STT", "Ten", "Ma"], ["1.", "CNTT", "7480201"]]
        rmeta = [
            {"is_header": True, "row_index": 0, "stt": "STT", "source_pages": [31], "is_cross_page_spliced": False},
            {"is_header": False, "row_index": 1, "stt": "1.", "source_pages": [31], "is_cross_page_spliced": False},
        ]
        blocks = [
            {"type": "heading", "content": "10.1. Thông tin về học phí", "section": "10.1. Thông tin về học phí", "page": 28, "source_pages": [28]},
            self._make_table_block(rows, rmeta, page=28),
        ]
        docs = blocks_to_docs(blocks)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["metadata"]["content_type"], "table")
        self.assertIn("10.1. Thông tin về học phí", docs[0]["content"])

    def test_12g_standalone_heading_flushed(self):
        """Nếu heading nằm cuối tài liệu không có text/table sau nó, được flush an toàn."""
        blocks = [
            {"type": "heading", "content": "11. Lời kết", "section": "11. Lời kết", "page": 40, "source_pages": [40]},
        ]
        docs = blocks_to_docs(blocks)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "11. Lời kết")

    def test_continuation_block_inherits_root_header(self):
        """Test 12e: continuation block inherits header from root block of the same chain."""
        root_block = self._make_table_block(
            rows=[["TT", "Nganh", "Ma"], ["1.", "CNTT", "7480201"]],
            row_meta=[{"is_header": True, "row_index": 0}, {"is_header": False, "row_index": 1}],
            page=2,
        )
        root_block["table_chain_id"] = "test_chain_1"
        root_block["table_id"] = "p2_t0"

        cont_block = self._make_table_block(
            rows=[["2.", "DTVT", "7480202"], ["3.", "Co dien tu", "7480203"]],
            row_meta=[{"is_header": False, "row_index": 0}, {"is_header": False, "row_index": 1}],
            page=3,
        )
        cont_block["table_chain_id"] = "test_chain_1"
        cont_block["table_id"] = "p3_t0"
        cont_block["is_continuation"] = True

        docs = blocks_to_docs([root_block, cont_block])
        self.assertEqual(len(docs), 2)

        # Root block has its header
        self.assertIn("| TT | Nganh | Ma |", docs[0]["content"])
        self.assertIn("| 1. | CNTT | 7480201 |", docs[0]["content"])

        # Continuation block inherits the root header
        self.assertIn("| TT | Nganh | Ma |", docs[1]["content"])
        self.assertIn("| 2. | DTVT | 7480202 |", docs[1]["content"])
        self.assertIn("| 3. | Co dien tu | 7480203 |", docs[1]["content"])

    def test_first_data_row_preserved_as_data_row(self):
        """Test 12f: first row of continuation block remains a data row, not a table header."""
        root_block = self._make_table_block(
            rows=[["STT", "Ten"], ["1.", "A"]],
            row_meta=[{"is_header": True}, {"is_header": False}],
            page=1,
        )
        root_block["table_chain_id"] = "chain_alpha"

        cont_block = self._make_table_block(
            rows=[["6.", "Nganh STT 6"]],
            row_meta=[{"is_header": False}],
            page=2,
        )
        cont_block["table_chain_id"] = "chain_alpha"
        cont_block["is_continuation"] = True

        docs = blocks_to_docs([root_block, cont_block])
        cont_content = docs[1]["content"]

        lines = [line.strip() for line in cont_content.split("\n") if line.strip()]
        separator_idx = next(i for i, l in enumerate(lines) if l.startswith("| --- |"))
        row_6_idx = next(i for i, l in enumerate(lines) if l.startswith("| 6. |"))
        self.assertLess(separator_idx, row_6_idx, "STT 6 must be a data row after separator, not header")

    def test_chain_isolation_different_headers(self):
        """Test 12g: two different chains inherit their own respective headers without crosstalk."""
        chain1_root = self._make_table_block(
            rows=[["TT", "Mon HSG"], ["1.", "Toan"]],
            row_meta=[{"is_header": True}, {"is_header": False}],
            page=2,
        )
        chain1_root["table_chain_id"] = "chain_hsg"

        chain1_cont = self._make_table_block(
            rows=[["2.", "Ly"]],
            row_meta=[{"is_header": False}],
            page=3,
        )
        chain1_cont["table_chain_id"] = "chain_hsg"
        chain1_cont["is_continuation"] = True

        chain2_root = self._make_table_block(
            rows=[["STT", "Nganh", "Hoc phi"], ["1.", "CNTT", "20tr"]],
            row_meta=[{"is_header": True}, {"is_header": False}],
            page=10,
        )
        chain2_root["table_chain_id"] = "chain_tuition"

        chain2_cont = self._make_table_block(
            rows=[["2.", "Kinh te", "18tr"]],
            row_meta=[{"is_header": False}],
            page=11,
        )
        chain2_cont["table_chain_id"] = "chain_tuition"
        chain2_cont["is_continuation"] = True

        docs = blocks_to_docs([chain1_root, chain1_cont, chain2_root, chain2_cont])

        # chain1_cont must inherit chain_hsg header, NOT chain_tuition header
        self.assertIn("| TT | Mon HSG |", docs[1]["content"])
        self.assertNotIn("Hoc phi", docs[1]["content"])

        # chain2_cont must inherit chain_tuition header, NOT chain_hsg header
        self.assertIn("| STT | Nganh | Hoc phi |", docs[3]["content"])
        self.assertNotIn("Mon HSG", docs[3]["content"])

    def test_multi_row_header_inherited(self):
        """Test 12h: multi-row headers are fully preserved and inherited in continuation blocks."""
        root = self._make_table_block(
            rows=[
                ["Nhom", "Chi tiet", "Chi tiet"],
                ["STT", "Nganh", "Diem chieu"],
                ["1.", "CNTT", "25.5"],
            ],
            row_meta=[
                {"is_header": True},
                {"is_header": True},
                {"is_header": False},
            ],
            page=1,
        )
        root["table_chain_id"] = "multi_row_chain"

        cont = self._make_table_block(
            rows=[["2.", "DTVT", "24.0"]],
            row_meta=[{"is_header": False}],
            page=2,
        )
        cont["table_chain_id"] = "multi_row_chain"
        cont["is_continuation"] = True

        docs = blocks_to_docs([root, cont])
        # Both header rows must be present in continuation block
        self.assertIn("| Nhom | Chi tiet | Chi tiet |", docs[1]["content"])
        self.assertIn("| STT | Nganh | Diem chieu |", docs[1]["content"])
        self.assertIn("| 2. | DTVT | 24.0 |", docs[1]["content"])

    def test_no_root_header_safe_fallback(self):
        """Test 12i: table with no header and no chain uses generic fallback headers; rows preserved."""
        standalone = self._make_table_block(
            rows=[["Data A1", "Data B1"], ["Data A2", "Data B2"]],
            row_meta=[{"is_header": False}, {"is_header": False}],
            page=1,
        )
        standalone["table_chain_id"] = None

        docs = blocks_to_docs([standalone])
        self.assertEqual(len(docs), 1)
        self.assertIn("| Cột 1 | Cột 2 |", docs[0]["content"])
        self.assertIn("| Data A1 | Data B1 |", docs[0]["content"])
        self.assertIn("| Data A2 | Data B2 |", docs[0]["content"])

    def test_continuation_subchunks_have_inherited_header(self):
        """Test 12j: sub-chunks produced from a split continuation block all contain inherited header."""
        root = self._make_table_block(
            rows=[["ColA", "ColB"], ["R0A", "R0B"]],
            row_meta=[{"is_header": True}, {"is_header": False}],
            page=1,
        )
        root["table_chain_id"] = "split_chain"

        # Large continuation block that will require multiple chunks
        long_data = [[f"D_{i:04d}" + "x" * 60, f"V_{i}"] for i in range(30)]
        cont = self._make_table_block(
            rows=long_data,
            row_meta=[{"is_header": False} for _ in range(30)],
            page=2,
        )
        cont["table_chain_id"] = "split_chain"
        cont["is_continuation"] = True

        docs = blocks_to_docs([root, cont])
        chunks = chunk_documents(docs)

        cont_chunks = [c for c in chunks if c["metadata"]["page"] == 2]
        self.assertGreater(len(cont_chunks), 1, "Should be split into multiple chunks")
        for chunk in cont_chunks:
            self.assertIn("| ColA | ColB |", chunk["content"])
            self.assertIn("Trang: 2", chunk["content"])

    def test_original_blocks_not_mutated(self):
        """Test 12k: original block dictionary and rows are not modified during conversion."""
        orig_rows = [["1.", "Val"]]
        orig_meta = [{"is_header": False}]
        cont = self._make_table_block(
            rows=orig_rows,
            row_meta=orig_meta,
            page=2,
        )
        cont["table_chain_id"] = "mut_chain"
        cont["is_continuation"] = True

        blocks = [
            self._make_table_block(
                rows=[["H1", "H2"], ["0.", "V0"]],
                row_meta=[{"is_header": True}, {"is_header": False}],
                page=1,
            ),
            cont,
        ]
        blocks[0]["table_chain_id"] = "mut_chain"

        _ = blocks_to_docs(blocks)

        self.assertEqual(cont["content"]["rows_normalized"], [["1.", "Val"]])
        self.assertEqual(len(cont["content"]["row_metadata"]), 1)
        self.assertFalse(cont["content"]["row_metadata"][0]["is_header"])


# ---------------------------------------------------------------------------
# Group D: Real PDF smoke tests
# ---------------------------------------------------------------------------

PDF_PATH = Path("data/dsk-thong-tin-tuyen-sinh-2026-new-0162026-25-ctdt-1950-ct.pdf")


class TestRealPDFSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not PDF_PATH.exists():
            cls.skip_pdf = True
            return
        cls.skip_pdf = False
        from app.rag.ingestion.pdf_extractor import extract_pdf
        res = extract_pdf(PDF_PATH)
        cls.blocks = res["blocks"]
        cls.docs = blocks_to_docs(cls.blocks)
        cls.chunks = chunk_documents(cls.docs)

    def _skip(self):
        if self.skip_pdf:
            self.skipTest("PDF not found")

    def test_hsg_table_prefix_and_header_present(self):
        """Test 13a: HSG table chunks have context prefix; header 'TT' appears
        in EVERY chunk (inherited from root block in continuation chunks)."""
        self._skip()
        hsg = [c for c in self.chunks if c["metadata"].get("table_chain_id") == "table_chain_1_p2"]
        self.assertGreater(len(hsg), 0)
        # Every chunk must have a context prefix with "Trang:"
        for c in hsg:
            self.assertIn("Trang:", c["content"])
            self.assertIn("| TT |", c["content"])
            self.assertIn("| Môn thi HSG |", c["content"])

        # Page 3 continuation block: verify STT 6 is a data row, not a header
        p3_chunks = [c for c in hsg if c["metadata"].get("page") == 3]
        self.assertGreater(len(p3_chunks), 0)
        p3_content = p3_chunks[0]["content"]
        self.assertIn("| 6. |", p3_content)
        lines = [l.strip() for l in p3_content.split("\n") if l.strip()]
        separator_idx = next(i for i, l in enumerate(lines) if l.startswith("| --- |"))
        row_6_idx = next(i for i, l in enumerate(lines) if l.startswith("| 6. |"))
        self.assertLess(separator_idx, row_6_idx, "STT 6 must be placed after separator as a data row")

    def test_chain8_all_stts_recoverable(self):
        """Test 13b: admissions chain 8 chunks collectively cover STT 1, 12, 24."""
        self._skip()
        c8 = [c for c in self.chunks if c["metadata"].get("table_chain_id") == "table_chain_8_p31"]
        self.assertGreater(len(c8), 0)
        all_text = "\n".join(c["content"] for c in c8)
        for stt in ["1.", "12.", "24."]:
            self.assertIn(stt, all_text, f"STT {stt} missing from admissions chunks")

    def test_no_empty_chunks_real_pdf(self):
        """Test 13c: no chunk from the real PDF is empty."""
        self._skip()
        for c in self.chunks:
            self.assertTrue(c["content"].strip(), "Got empty chunk")

    def test_chunk_index_monotonic(self):
        """Test 13d: global chunk_index is 0-based and gap-free."""
        self._skip()
        indices = [c["chunk_index"] for c in self.chunks]
        self.assertEqual(indices, list(range(len(indices))))

    def test_metadata_json_serialisable(self):
        """Test 13e: all chunk metadata is JSON-serialisable."""
        self._skip()
        for c in self.chunks:
            try:
                json.dumps(c["metadata"])
            except Exception as e:
                self.fail(f"Not JSON-serialisable: {e}")

    def test_table_chunks_have_table_id_and_section(self):
        """Test 13f: table chunks carry table_id and section metadata."""
        self._skip()
        tbl_chunks = [c for c in self.chunks if c["metadata"]["content_type"] == "table"]
        self.assertGreater(len(tbl_chunks), 0)
        for c in tbl_chunks:
            self.assertIsNotNone(c["metadata"].get("table_id"))
            self.assertIsNotNone(c["metadata"].get("section"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

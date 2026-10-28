from __future__ import annotations

import logging
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 1. Adapter: extract_pdf() blocks -> doc list
# ──────────────────────────────────────────────────────────────────

def _extract_chain_headers(blocks: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    """
    Xác định header gốc cho từng table_chain_id từ block đầu tiên của chuỗi có header hợp lệ.
    Bảo toàn cấu trúc header nhiều tầng (nếu có).
    Không làm thay đổi rows_raw hay cấu trúc dữ liệu gốc của blocks.
    """
    chain_headers: dict[str, list[list[str]]] = {}
    for block in blocks:
        if block.get('type') != 'table':
            continue
        chain_id = block.get('table_chain_id')
        if not chain_id or chain_id in chain_headers:
            continue

        raw_content = block.get('content', {})
        rows = raw_content.get('rows_normalized', [])
        row_meta = raw_content.get('row_metadata', [])

        if not rows or not row_meta:
            continue

        headers = []
        for row, meta in zip(rows, row_meta):
            if meta.get('is_header'):
                headers.append(list(row))

        if headers:
            chain_headers[chain_id] = headers

    return chain_headers


def _rows_to_markdown(
    rows: list[list[str]],
    row_metadata: list[dict[str, Any]] | None = None,
    inherited_headers: list[list[str]] | None = None,
) -> str:
    if not rows and not inherited_headers:
        return ''

    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []

    if row_metadata:
        for row, meta in zip(rows, row_metadata):
            if meta.get('is_header'):
                header_rows.append(list(row))
            else:
                data_rows.append(list(row))
    else:
        data_rows = [list(r) for r in rows]

    # 1. Kế thừa header từ chuỗi bảng nếu block này không có header riêng
    if not header_rows and inherited_headers:
        header_rows = [list(hr) for hr in inherited_headers]
        # data_rows giữ nguyên toàn bộ rows hiện tại (không lấy rows[0] làm header)

    # 2. Xử lý an toàn khi hoàn toàn không có header hợp lệ:
    # Không tự ý lấy hàng dữ liệu đầu tiên làm header; áp dụng generic column names
    if not header_rows:
        col_count = max((len(r) for r in data_rows), default=1)
        header_rows = [[f'Cột {i + 1}' for i in range(col_count)]]
        logger.warning(
            'Bảng không có header hợp lệ và không có header kế thừa từ chain. '
            'Áp dụng fallback generic headers (%s), bảo toàn toàn bộ hàng dữ liệu.',
            header_rows[0],
        )

    col_count = max((len(r) for r in header_rows + data_rows), default=1)

    def render_row(r: list[str]) -> str:
        cells = list(r)
        while len(cells) < col_count:
            cells.append('')
        cells = cells[:col_count]
        cells = [c.replace('\n', ' ') for c in cells]
        return '| ' + ' | '.join(cells) + ' |'

    lines = []
    for hr in header_rows:
        lines.append(render_row(hr))
    lines.append('| ' + ' | '.join(['---'] * col_count) + ' |')
    for dr in data_rows:
        lines.append(render_row(dr))

    return '\n'.join(lines)


def blocks_to_docs(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    chain_headers = _extract_chain_headers(blocks)
    pending_headings: list[dict[str, Any]] = []

    def _flush_pending_headings() -> None:
        """Nếu heading nằm ở cuối tài liệu hoặc không có block nội dung theo sau, emit an toàn."""
        nonlocal pending_headings
        if not pending_headings:
            return
        combined_text = "\n\n".join(h["content"] for h in pending_headings if h["content"].strip())
        if combined_text.strip():
            first_h = pending_headings[0]
            all_pages = sorted(list(set(
                p for h in pending_headings for p in h.get("source_pages", [h.get("page", 0)]) if p
            )))
            docs.append({
                "content": combined_text,
                "metadata": {
                    "content_type": "text",
                    "page": first_h.get("page"),
                    "source_pages": all_pages or [first_h.get("page", 0)],
                    "section": first_h.get("section", ""),
                    "table_id": None,
                    "table_chain_id": None,
                },
            })
        pending_headings = []

    for block in blocks:
        btype = block.get('type', 'text')
        raw_content = block.get('content', '')
        section = block.get('section', '')
        source_pages = block.get('source_pages', [block.get('page', 0)])

        if btype == 'heading':
            heading_str = raw_content if isinstance(raw_content, str) else str(raw_content)
            if heading_str.strip():
                pending_headings.append({
                    "content": heading_str.strip(),
                    "page": block.get('page'),
                    "source_pages": source_pages,
                    "section": section,
                })
            continue

        if btype == 'table':
            rows = raw_content.get('rows_normalized', [])
            row_meta = raw_content.get('row_metadata', [])
            if not rows:
                continue

            chain_id = block.get('table_chain_id')
            inherited = chain_headers.get(chain_id) if chain_id else None

            md_table = _rows_to_markdown(rows, row_meta, inherited_headers=inherited)
            if not md_table.strip():
                continue

            # Kết hợp trang nguồn và tiêu đề từ pending headings
            table_pages = list(source_pages)
            heading_prefix = ""
            if pending_headings:
                heading_lines = [h["content"] for h in pending_headings if h["content"]]
                for h in pending_headings:
                    for hp in h.get("source_pages", []):
                        if hp and hp not in table_pages:
                            table_pages.append(hp)
                table_pages.sort()
                pending_headings = []
                # Nếu heading chưa xuất hiện trọn vẹn trong section, đặt ở đầu prefix bảng
                unmatched = [hl for hl in heading_lines if hl not in section]
                if unmatched:
                    heading_prefix = "\n\n".join(unmatched) + "\n\n"

            pages_str = ', '.join(str(p) for p in table_pages)
            context_prefix = f'Phan: {section} | Trang: {pages_str}'
            content = f'{heading_prefix}{context_prefix}\n\n{md_table}'.strip()

            metadata = {
                'content_type': 'table',
                'page': block.get('page'),
                'source_pages': table_pages,
                'section': section,
                'table_id': block.get('table_id'),
                'table_chain_id': chain_id,
            }
        else:
            text_str = raw_content if isinstance(raw_content, str) else str(raw_content)
            if not text_str.strip():
                continue

            text_pages = list(source_pages)
            if pending_headings:
                heading_prefixes = [h["content"] for h in pending_headings if h["content"]]
                for h in pending_headings:
                    for hp in h.get("source_pages", []):
                        if hp and hp not in text_pages:
                            text_pages.append(hp)
                text_pages.sort()
                content = "\n\n".join(heading_prefixes + [text_str.strip()])
                pending_headings = []
            else:
                content = text_str.strip()

            metadata = {
                'content_type': 'text',
                'page': block.get('page'),
                'source_pages': text_pages,
                'section': section,
                'table_id': None,
                'table_chain_id': None,
            }

        docs.append({'content': content, 'metadata': metadata})

    _flush_pending_headings()
    return docs


# ──────────────────────────────────────────────────────────────────
# 2. Table chunker
# ──────────────────────────────────────────────────────────────────

_SEP_RE = re.compile(r'^\|[\s|:\-]+\|$')


def split_markdown_table(content, max_chunk_chars=1200):
    if not content or not content.strip():
        return [content] if content else []

    raw_lines = content.split('\n')

    prefix_lines = []
    table_lines = []
    in_table = False
    for line in raw_lines:
        stripped = line.strip()
        if not in_table:
            if stripped.startswith('|'):
                in_table = True
                table_lines.append(line)
            else:
                prefix_lines.append(line)
        else:
            table_lines.append(line)

    if not table_lines:
        return [content]

    header_rows = []
    separator_line = None
    data_rows = []

    for line in table_lines:
        stripped = line.strip()
        if separator_line is None:
            if stripped.startswith('|') and _SEP_RE.match(stripped):
                separator_line = line
            elif stripped.startswith('|'):
                header_rows.append(line)
        else:
            if stripped.startswith('|') or stripped:
                data_rows.append(line)

    if not header_rows or separator_line is None:
        return [content]

    if len(content) <= max_chunk_chars:
        return [content]

    if not data_rows:
        return [content]

    base_prefix = '\n'.join(prefix_lines).strip()
    header_block = '\n'.join(header_rows) + '\n' + separator_line

    def _make_chunk(batch):
        body = '\n'.join(batch)
        if base_prefix:
            return f'{base_prefix}\n\n{header_block}\n{body}'
        return f'{header_block}\n{body}'

    chunks = []
    current_batch = []
    overhead = len(base_prefix) + 2 + len(header_block) + 1
    current_size = overhead

    for row in data_rows:
        row_len = len(row) + 1

        if overhead + row_len > max_chunk_chars and not current_batch:
            chunks.append(_make_chunk([row]))
            current_size = overhead
            continue

        if current_batch and current_size + row_len > max_chunk_chars:
            chunks.append(_make_chunk(current_batch))
            current_batch = [row]
            current_size = overhead + row_len
        else:
            current_batch.append(row)
            current_size += row_len

    if current_batch:
        chunks.append(_make_chunk(current_batch))

    return chunks if chunks else [content]


# ──────────────────────────────────────────────────────────────────
# 3. Main chunking entry point
# ──────────────────────────────────────────────────────────────────

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=['\n\n', '\n', '. ', '; ', ', ', ' ', ''],
        length_function=len,
    )

    chunks = []
    global_counter = 0

    for doc in docs:
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})

        if not content.strip():
            continue

        c_type = metadata.get('content_type', 'text')

        if c_type == 'table':
            parts = split_markdown_table(content, max_chunk_chars=1200)
        else:
            parts = text_splitter.split_text(content)

        for doc_idx, part in enumerate(parts):
            if not part.strip():
                continue
            chunks.append({
                'chunk_index': global_counter,
                'doc_chunk_index': doc_idx,
                'content': part,
                'metadata': {**metadata},
            })
            global_counter += 1

    return chunks

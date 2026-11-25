import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pypdf

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional at runtime in some envs
    pdfplumber = None

from app.core.config import (
    DOCUMENT_NAME,
    DOCUMENT_VERSION,
    ACADEMIC_YEAR,
)

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+){0,4}[.)]?\s+)?[A-Z][A-Z0-9 ,&()/'\-:]{3,}$")
_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+){1,5}|\([a-zA-Z0-9]+\)|[a-z]\)|[ivx]+\.)\s+")
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{2,4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,?\s+\d{2,4})?)\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _norm_space(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 4 or len(s) > 120:
        return False
    if _CLAUSE_RE.match(s):
        return False
    if s.endswith(('.', ';')):
        return False
    if _HEADING_RE.match(s) and sum(ch.isalpha() for ch in s) >= 4:
        # avoid all-caps fragments that are mostly numbers/symbols
        letters = [c for c in s if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / max(len(letters), 1) >= 0.7:
            return True
    # Title Case short lines can also be headings in this handbook
    if len(s.split()) <= 10 and s[:1].isupper() and not s.endswith(','):
        words = [w for w in re.split(r"\s+", s) if any(ch.isalpha() for ch in w)]
        if words and sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.8 and len(s) <= 80:
            return True
    return False


def _table_to_text(table: List[List[Optional[str]]], title: Optional[str] = None) -> str:
    if not table:
        return ""
    rows = [[_norm_space(cell or "") for cell in row] for row in table]
    # drop fully empty rows
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    norm_rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # detect header row: first non-empty row with multiple filled cells
    header_idx = 0
    for i, r in enumerate(norm_rows[:2]):
        if sum(1 for c in r if c) >= max(2, max_cols // 2):
            header_idx = i
            break
    headers = norm_rows[header_idx]
    headers = [h if h else f"Column {i+1}" for i, h in enumerate(headers)]

    lines = []
    if title:
        lines.append(f"Table: {title}")
    lines.append("Columns: " + " | ".join(h for h in headers if h))

    for ri, row in enumerate(norm_rows):
        if ri == header_idx and any(row):
            continue
        if not any(row):
            continue
        # If first column looks like a row label, preserve it explicitly
        row_label = row[0] if row and row[0] else ""
        parts = []
        for ci, cell in enumerate(row):
            if not cell:
                continue
            col = headers[ci] if ci < len(headers) else f"Column {ci+1}"
            parts.append(f"{col}: {cell}")
        if parts:
            prefix = f"Row {ri - header_idx if ri > header_idx else ri + 1}"
            if row_label and max_cols > 1:
                prefix = f"{row_label}"
            lines.append(f"- {prefix} -> " + "; ".join(parts))

    # fallback pipe table if parsing was poor
    if len(lines) <= 2:
        lines = []
        if title:
            lines.append(f"Table: {title}")
        for row in norm_rows:
            if any(row):
                lines.append(" | ".join(cell if cell else "-" for cell in row))
    return "\n".join(lines).strip()


def _table_to_records(table: List[List[Optional[str]]], meta: Dict[str, Any], title: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [[_norm_space(cell or "") for cell in row] for row in table if any(_norm_space(cell or "") for cell in row)]
    if not rows:
        return []
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    header_idx = 0
    for i, r in enumerate(rows[:2]):
        if sum(1 for c in r if c) >= max(2, max_cols // 2):
            header_idx = i
            break
    headers = [h if h else f"Column {i+1}" for i, h in enumerate(rows[header_idx])]
    records = []
    for ri, row in enumerate(rows):
        if ri == header_idx:
            continue
        if not any(row):
            continue
        rec = {headers[i]: row[i] for i in range(min(len(headers), len(row))) if row[i]}
        if not rec:
            continue
        row_label = row[0] if row[0] else None
        record_text = "; ".join(f"{k}: {v}" for k, v in rec.items())
        records.append({
            **meta,
            "content_type": "table_record",
            "table_name": title or meta.get("section") or "Table",
            "row_label": row_label,
            "record": rec,
            "text": f"Table: {title or meta.get('section') or 'Table'} | Page {meta.get('page')} | {record_text}",
        })
    return records


def _ocr_page(pdf_path: str, page_index_zero_based: int) -> str:
    """Best-effort OCR fallback. Returns empty string if OCR deps/tools unavailable."""
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""
    try:
        images = convert_from_path(pdf_path, first_page=page_index_zero_based + 1, last_page=page_index_zero_based + 1)
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0]) or ""
        return _norm_space(text)
    except Exception as e:
        logger.warning("OCR fallback failed on page %s: %s", page_index_zero_based + 1, e)
        return ""


def _extract_pages_pypdf(pdf_path: str) -> Dict[int, str]:
    reader = pypdf.PdfReader(pdf_path)
    out: Dict[int, str] = {}
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        out[i + 1] = _norm_space(raw)
    return out


def _detect_section_hierarchy(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    section = subsection = clause = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _CLAUSE_RE.match(line)
        if m:
            token = m.group(1)
            depth = token.count('.') + 1 if token[0].isdigit() else 3
            if depth <= 1:
                section = line[:120]
            elif depth == 2:
                subsection = line[:120]
            clause = line[:160]
            continue
        if _is_heading(line):
            # promote previous section down when a new top-like heading appears after content
            if section and not subsection:
                subsection = line[:120]
            elif section and subsection and len(line) <= 60:
                # keep as subsub heading context in clause field
                clause = line[:160]
            else:
                section = line[:120]
                subsection = None
                clause = None
    return section, subsection, clause


def _classify_content(text: str, has_table: bool = False) -> str:
    t = text.lower()
    if has_table:
        return "table"
    if any(k in t for k in ["calendar", "semester", "trimester", "commencement", "examination schedule", "holiday", "academic calendar"]):
        if _DATE_RE.search(text):
            return "academic_calendar"
    if _DATE_RE.search(text) and any(k in t for k in ["date", "semester", "exam", "registration", "commence", "last day", "holiday"]):
        return "date_event"
    if any(k in t for k in ["eligibility", "eligible", "criteria", "minimum", "required to pass", "aggregate", "attendance", "arrears"]):
        return "eligibility"
    if any(k in t for k in ["fee", "fees", "fine", "penalty", "borrow", "loan period", "timing", "hours", "amount", "rupees", "rs."]):
        if _NUM_RE.search(text):
            return "numeric_policy"
    if any(k in t for k in ["regulation", "policy", "code of conduct", "discipline", "misconduct", "grievance", "procedure", "shall", "must", "may"]):
        return "policy"
    return "text"


def extract_document(pdf_path: str) -> Dict[str, Any]:
    """Extract page text, tables, and structured records with metadata."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Handbook PDF not found at {pdf_path}")

    logger.info("Extracting handbook from %s", pdf_path)
    pypdf_pages = _extract_pages_pypdf(pdf_path)

    pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    table_records: List[Dict[str, Any]] = []
    ocr_pages: List[int] = []
    empty_pages: List[int] = []

    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    plumber_text = _norm_space(page.extract_text() or "")
                    fallback_text = pypdf_pages.get(page_num, "")
                    text = plumber_text if len(plumber_text) >= len(fallback_text) else fallback_text

                    # OCR fallback for pages with very little/no extractable text
                    if len(text.strip()) < 40:
                        ocr_text = _ocr_page(pdf_path, i)
                        if ocr_text and len(ocr_text) > len(text):
                            text = ocr_text
                            ocr_pages.append(page_num)

                    if len(text.strip()) < 20:
                        empty_pages.append(page_num)

                    extracted_tables = []
                    try:
                        extracted_tables = page.extract_tables() or []
                    except Exception as e:
                        logger.warning("Table extraction failed on page %s: %s", page_num, e)

                    page_tables = []
                    for ti, tbl in enumerate(extracted_tables):
                        # identify a possible title from nearby text lines (best-effort)
                        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                        title = None
                        # heuristic: heading-like line before table likely not directly known; use page/section context later
                        if lines:
                            for ln in lines[:8]:
                                if _is_heading(ln) and len(ln) <= 80:
                                    title = ln
                        tbl_text = _table_to_text(tbl, title=title)
                        if not tbl_text:
                            continue
                        meta = {
                            "document": DOCUMENT_NAME,
                            "document_version": DOCUMENT_VERSION,
                            "academic_year": ACADEMIC_YEAR,
                            "page": page_num,
                            "table_index": ti,
                        }
                        recs = _table_to_records(tbl, meta, title=title)
                        page_tables.append({
                            **meta,
                            "table_name": title or f"Page {page_num} Table {ti+1}",
                            "text": tbl_text,
                            "row_count": len(tbl),
                        })
                        table_records.extend(recs)

                    pages.append({
                        "page": page_num,
                        "text": text,
                        "tables": page_tables,
                    })
                    tables.extend(page_tables)
        except Exception as e:
            logger.exception("pdfplumber extraction failed, falling back to pypdf-only: %s", e)
            pdfplumber_failed = True
    else:
        logger.warning("pdfplumber not installed; using pypdf-only extraction (tables/structure may be limited).")

    if not pages:
        # pypdf-only fallback
        for page_num, text in pypdf_pages.items():
            if len(text.strip()) < 40:
                ocr_text = _ocr_page(pdf_path, page_num - 1)
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
                    ocr_pages.append(page_num)
            if len(text.strip()) < 20:
                empty_pages.append(page_num)
            pages.append({"page": page_num, "text": text, "tables": []})

    if empty_pages:
        logger.warning("Pages with little/no extractable text after fallback: %s", empty_pages[:20])
    if ocr_pages:
        logger.info("OCR fallback used on pages: %s", ocr_pages)

    # Extract dedicated high-accuracy academic calendar records
    cal_chunks = []
    try:
        from app.rag.calendar_parser import extract_calendar_records
        cal_chunks, cal_records = extract_calendar_records(pdf_path)
        table_records.extend(cal_records)
    except Exception as e:
        logger.warning("Could not extract dedicated calendar records: %s", e)
        cal_chunks = []

    return {
        "document": DOCUMENT_NAME,
        "document_version": DOCUMENT_VERSION,
        "academic_year": ACADEMIC_YEAR,
        "pages": pages,
        "tables": tables,
        "table_records": table_records,
        "calendar_chunks": cal_chunks,
        "stats": {
            "page_count": len(pages),
            "table_count": len(tables),
            "table_record_count": len(table_records),
            "ocr_pages": ocr_pages,
            "empty_pages": empty_pages,
        },
    }

def _split_into_blocks(text: str) -> List[Dict[str, Any]]:
    """Split page text into heading/paragraph/clause-aware blocks."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    blocks: List[Dict[str, Any]] = []
    buf: List[str] = []
    section = subsection = clause = None

    def flush(kind: str = "paragraph"):
        nonlocal buf
        content = _norm_space("\n".join(buf))
        if content:
            blocks.append({
                "kind": kind,
                "text": content,
                "section": section,
                "subsection": subsection,
                "clause": clause,
            })
        buf = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush("paragraph")
            continue
        if _is_heading(line):
            flush("paragraph")
            if section and not subsection:
                subsection = line[:120]
            elif section and subsection:
                # keep deeper heading as clause/context marker but start new semantic block
                clause = line[:160]
                subsection = subsection
            else:
                section = line[:120]
                subsection = None
                clause = None
            blocks.append({
                "kind": "heading",
                "text": line[:160],
                "section": section,
                "subsection": subsection,
                "clause": clause,
            })
            continue
        if _CLAUSE_RE.match(line):
            # clause often starts a new regulation unit; keep with following text
            if buf:
                flush("paragraph")
            clause = line[:160]
            buf.append(line)
            continue
        buf.append(line)
    flush("paragraph")

    # merge tiny adjacent paragraph blocks under same context
    merged: List[Dict[str, Any]] = []
    for b in blocks:
        if b["kind"] == "heading":
            merged.append(b)
            continue
        if (
            merged
            and merged[-1]["kind"] == "paragraph"
            and merged[-1]["section"] == b["section"]
            and merged[-1]["subsection"] == b["subsection"]
            and len(merged[-1]["text"]) < 350
            and len(b["text"]) < 350
        ):
            merged[-1]["text"] = _norm_space(merged[-1]["text"] + "\n" + b["text"])
            if b.get("clause"):
                merged[-1]["clause"] = b["clause"]
        else:
            merged.append(b)
    return merged


def _pack_blocks(blocks: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[str]:
    """Pack semantic blocks into chunks respecting boundaries and target size."""
    texts: List[str] = []
    current: List[str] = []
    cur_len = 0

    def emit():
        nonlocal current, cur_len
        if not current:
            return
        chunk_text = _norm_space("\n".join(current))
        if chunk_text:
            texts.append(chunk_text)
        if overlap > 0 and chunk_text:
            tail = chunk_text[-overlap:]
            # try to start overlap at sentence boundary
            m = re.search(r"[.!?]\s+", tail)
            if m:
                tail = tail[m.end():].strip()
            current = [tail] if tail else []
            cur_len = len(tail)
        else:
            current = []
            cur_len = 0

    for b in blocks:
        t = b["text"].strip()
        if not t:
            continue
        # keep headings attached to following content where possible
        if b["kind"] == "heading":
            if cur_len > chunk_size * 0.7:
                emit()
            current.append(t)
            cur_len += len(t) + 1
            continue

        # split very long paragraph text on sentence-ish boundaries
        parts = [t]
        if len(t) > chunk_size:
            sents = re.split(r"(?<=[.!?])\s+|\n+", t)
            parts, buf, blen = [], "", 0
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if blen + len(s) + 1 <= chunk_size:
                    buf = (buf + " " + s).strip() if buf else s
                    blen = len(buf)
                else:
                    if buf:
                        parts.append(buf)
                    # hard wrap oversized sentence
                    while len(s) > chunk_size:
                        parts.append(s[:chunk_size].strip())
                        s = s[chunk_size - overlap:chunk_size - overlap + chunk_size].strip() if overlap < chunk_size else s[chunk_size:].strip()
                    buf = s
                    blen = len(s)
            if buf:
                parts.append(buf)

        for part in parts:
            if not part:
                continue
            if cur_len + len(part) + 1 > chunk_size and current:
                emit()
            current.append(part)
            cur_len += len(part) + 1
            # if a single block filled chunk heavily, emit to keep chunks bounded
            if cur_len >= chunk_size:
                emit()
    emit()

    # merge tiny trailing chunks
    final: List[str] = []
    for t in texts:
        if final and len(t) < 220 and len(final[-1]) + len(t) < chunk_size + 100:
            final[-1] = _norm_space(final[-1] + "\n" + t)
        else:
            final.append(t)
    return final


def build_chunks(extracted: Dict[str, Any], chunk_size: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:
    """Build metadata-rich, semantically chunked records from extracted document."""
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page_obj in extracted.get("pages", []):
        page_num = page_obj.get("page")
        text = page_obj.get("text") or ""
        base_meta = {
            "document": extracted.get("document", DOCUMENT_NAME),
            "document_version": extracted.get("document_version", DOCUMENT_VERSION),
            "academic_year": extracted.get("academic_year", ACADEMIC_YEAR),
            "page": page_num,
            "campus": None,
            "programme": None,
        }

        # 1) Table chunks first (kept understandable independently)
        for tbl in page_obj.get("tables", []):
            tbl_text = (tbl.get("text") or "").strip()
            if not tbl_text:
                continue
            section = tbl.get("table_name") or f"Page {page_num} Table"
            content_type = "table"
            chunk = {
                "chunk_id": chunk_id,
                "id": f"p{page_num}-tbl-{tbl.get('table_index', 0)}",
                **base_meta,
                "section": section,
                "subsection": None,
                "clause": None,
                "content_type": content_type,
                "table_name": tbl.get("table_name"),
                "text": tbl_text,
            }
            chunk["search_text"] = _build_search_text(chunk)
            chunks.append(chunk)
            chunk_id += 1

        # 2) Page text blocks with hierarchy-aware chunking
        blocks = _split_into_blocks(text)
        if not blocks and not text.strip():
            continue

        # infer page-level dominant section from headings/clauses
        page_section, page_subsection, _ = _detect_section_hierarchy(text.splitlines())

        packed = _pack_blocks(blocks, chunk_size=chunk_size, overlap=overlap) if blocks else []
        if not packed and text.strip():
            packed = [text.strip()[:chunk_size]]

        for idx, ctext in enumerate(packed):
            sec, subsec, clause = _detect_section_hierarchy(ctext.splitlines())
            section = sec or page_section
            subsection = subsec or page_subsection
            content_type = _classify_content(ctext, has_table=False)
            # detect campus/programme hints lightly for metadata filtering
            campus = _extract_campus_hint(ctext)
            programme = _extract_programme_hint(ctext)
            chunk = {
                "chunk_id": chunk_id,
                "id": f"p{page_num}-txt-{idx}",
                **base_meta,
                "section": section,
                "subsection": subsection,
                "clause": clause,
                "content_type": content_type,
                "table_name": None,
                "campus": campus,
                "programme": programme,
                "text": ctext.strip(),
            }
            chunk["search_text"] = _build_search_text(chunk)
            # skip duplicate/near-empty
            if len(chunk["text"]) < 40:
                continue
            chunks.append(chunk)
            chunk_id += 1

    # 3) Structured table records as searchable chunks/records
    for ri, rec in enumerate(extracted.get("table_records", [])):
        content_type = _classify_content(rec.get("text", ""), has_table=True)
        if _DATE_RE.search(rec.get("text", "")):
            content_type = "academic_calendar" if "calendar" in (rec.get("table_name") or "").lower() else content_type
        chunk = {
            "chunk_id": chunk_id,
            "id": f"p{rec.get('page')}-rec-{ri}",
            "document": rec.get("document", extracted.get("document", DOCUMENT_NAME)),
            "document_version": rec.get("document_version", extracted.get("document_version", DOCUMENT_VERSION)),
            "academic_year": rec.get("academic_year", extracted.get("academic_year", ACADEMIC_YEAR)),
            "page": rec.get("page"),
            "section": rec.get("table_name"),
            "subsection": rec.get("row_label"),
            "clause": None,
            "campus": _extract_campus_hint(rec.get("text", "")),
            "programme": _extract_programme_hint(rec.get("text", "")),
            "content_type": "table_record" if content_type == "table" else content_type,
            "table_name": rec.get("table_name"),
            "record": rec.get("record"),
            "text": rec.get("text", "").strip(),
        }
        chunk["search_text"] = _build_search_text(chunk)
        if chunk["text"]:
            chunks.append(chunk)
            chunk_id += 1

    # 4) Dedicated high-accuracy academic calendar chunks
    for cal_c in extracted.get("calendar_chunks", []):
        cal_c["chunk_id"] = chunk_id
        cal_c.setdefault("search_text", _build_search_text(cal_c))
        chunks.append(cal_c)
        chunk_id += 1

    logger.info("Built %s chunks (pages=%s, tables=%s, records=%s, calendar=%s)",
                len(chunks), len(extracted.get("pages", [])), len(extracted.get("tables", [])), len(extracted.get("table_records", [])), len(extracted.get("calendar_chunks", [])))
    return chunks


def _build_search_text(chunk: Dict[str, Any]) -> str:
    parts = [
        chunk.get("document") or "",
        chunk.get("section") or "",
        chunk.get("subsection") or "",
        chunk.get("clause") or "",
        chunk.get("table_name") or "",
        chunk.get("campus") or "",
        chunk.get("programme") or "",
        chunk.get("content_type") or "",
        chunk.get("text") or "",
    ]
    return _norm_space(" \n ".join(p for p in parts if p))


_CAMPUS_HINTS = [
    "bangalore", "bengaluru", "kengeri", "bannerghatta", "yeshwanthpur", "pune", "lavasa",
    "delhi", "ncr", "ghaziabad",
]


def _extract_campus_hint(text: str) -> Optional[str]:
    t = text.lower()
    for c in _CAMPUS_HINTS:
        if c in t:
            # normalize some variants
            if c == "bengaluru":
                return "Bangalore"
            return c.title()
    return None


def _extract_programme_hint(text: str) -> Optional[str]:
    t = text
    m = re.search(r"\b(B\.?Tech|M\.?Tech|BBA|MBA|BCom|MCom|BCA|MCA|BA|MA|BSc|MSc|LLB|LLM|Ph\.?D|PG|UG)\b", t, re.IGNORECASE)
    return m.group(1).upper().replace('.', '') if m else None


# Backward-compatible wrappers
def extract_handbook_pages(pdf_path: str) -> List[Dict[str, Any]]:
    extracted = extract_document(pdf_path)
    return [{"page": p["page"], "text": p.get("text", "")} for p in extracted.get("pages", [])]


def chunk_pages(pages_data: List[Dict[str, Any]], chunk_size: int = 900, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    extracted = {
        "document": DOCUMENT_NAME,
        "document_version": DOCUMENT_VERSION,
        "academic_year": ACADEMIC_YEAR,
        "pages": [{"page": p.get("page"), "text": p.get("text", ""), "tables": []} for p in pages_data],
        "tables": [],
        "table_records": [],
    }
    return build_chunks(extracted, chunk_size=chunk_size, overlap=chunk_overlap)

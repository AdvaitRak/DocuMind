# src/ingest.py

import os
import pickle
import tiktoken
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from rank_bm25 import BM25Okapi
from src.metrics import StageTimer, TokenUsage
import re
from datetime import datetime, timezone
import pymupdf as fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME   = "documind_chunks"
MAX_PAGES         = int(os.getenv("MAX_PDF_PAGES", 20))

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}
OCR_MIN_CHARS        = 30    
MAX_OCR_PAGES        = 10   
DAILY_OCR_PAGE_LIMIT = 100   
_ocr_daily = {"day": None, "pages": 0}
# ── Embeddings ────────────────────────────────────────────────────────────────

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# ── Splitter ──────────────────────────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# ── Extractors and helper functions (extract_pdf is main)────────────────────────────────────────────────────────────────

def _meta(file_path, page, session_id, **extra):
    m = {
        "source":     os.path.basename(file_path),
        "page":       page,
        "session_id": session_id,
    }
    m.update(extra)
    return m


def rows_to_markdown(rows) -> str:
    """List of rows (cells may be None) -> markdown table. '' if it isn't a real table."""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    if width < 2:
        return ""
    rows = [r + [""] * (width - len(r)) for r in rows]

    esc  = lambda s: s.replace("|", "\\|")
    line = lambda r: "| " + " | ".join(esc(c) for c in r) + " |"
    out  = [line(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
    out += [line(r) for r in rows[1:]]
    return "\n".join(out)


def split_table(md: str, page: int, max_chars: int = 1500) -> list[str]:
    """Keep small tables whole. Split big ones by rows, repeating the header in every part."""
    lines  = md.split("\n")
    header = lines[:2]
    body   = lines[2:]
    prefix = f"Table (page {page}):\n"

    if len(md) <= max_chars:
        return [prefix + md]

    parts, cur, size = [], [], 0
    for row in body:
        if cur and size + len(row) > max_chars:
            parts.append(prefix + "\n".join(header + cur))
            cur, size = [], 0
        cur.append(row)
        size += len(row) + 1
    if cur:
        parts.append(prefix + "\n".join(header + cur))
    return parts

def _reserve_ocr_pages(n: int):
    today = datetime.now(timezone.utc).date()
    if _ocr_daily["day"] != today:
        _ocr_daily["day"], _ocr_daily["pages"] = today, 0
    if _ocr_daily["pages"] + n > DAILY_OCR_PAGE_LIMIT:
        raise ValueError(
            "The daily OCR limit has been reached. Try again tomorrow, "
            "or upload a PDF with selectable text."
        )
    _ocr_daily["pages"] += n


def _page_ranges(nums: list[int]) -> str:
    """[1, 2, 3, 5] -> '1-3,5' (the format Azure expects)."""
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


def azure_ocr_pages(file_path: str, page_numbers: list[int]) -> dict[int, str]:
    """OCR only the given 1-based pages with Azure prebuilt-read. Returns {page: text}."""
    endpoint, key = os.getenv("DOCINTEL_ENDPOINT"), os.getenv("DOCINTEL_KEY")
    if not endpoint or not key:
        print("  OCR skipped: DOCINTEL_ENDPOINT / DOCINTEL_KEY not set")
        return {}
    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
    with open(file_path, "rb") as f:
        result = client.begin_analyze_document(
            "prebuilt-read",
            f,
            pages=_page_ranges(page_numbers),
            content_type="application/octet-stream",
        ).result()
    return {
        p.page_number: "\n".join(line.content for line in (p.lines or []))
        for p in result.pages
    }

def extract_pdf(file_path: str, session_id: str) -> list[Document]:
    doc = fitz.open(file_path)

    
    if len(doc) > MAX_PAGES:
        raise ValueError(f"Document has {len(doc)} pages. Maximum allowed is {MAX_PAGES}.")

    out = []
    scanned = []    
    fallback = {}    

    for i, page in enumerate(doc):
        page_no = i + 1
        table_rects, tables_md = [], []

        try:
            for tab in page.find_tables().tables:
                md = rows_to_markdown(table_rows(page, tab))
                if md:
                    table_rects.append(fitz.Rect(tab.bbox))
                    tables_md.append(md)
        except Exception as e:
            print(f"  table detection failed on page {page_no}: {e}")

       
        parts = []
        for b in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, txt, _, btype = b[:7]
            if btype != 0 or not txt.strip():
                continue
            center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(center in r for r in table_rects):
                continue
            parts.append(txt.strip())
        text = "\n".join(parts).strip()

        if len(text) < OCR_MIN_CHARS and not tables_md and page.get_images():
            scanned.append(page_no)
            if text:
                fallback[page_no] = text
        elif text:
            out.append(Document(page_content=text,
                                metadata=_meta(file_path, page_no, session_id)))

        for md in tables_md:
            out.append(Document(page_content=md,
                                metadata=_meta(file_path, page_no, session_id,
                                               content_type="table")))
    doc.close()

    if scanned:
        if len(scanned) > MAX_OCR_PAGES:
            raise ValueError(
                f"This looks like a scanned document ({len(scanned)} pages need OCR). "
                f"The limit is {MAX_OCR_PAGES} pages. Please upload a shorter file."
            )
        _reserve_ocr_pages(len(scanned))
        try:
            ocr_text = azure_ocr_pages(file_path, scanned)
        except Exception as e:
            print(f"  Azure OCR failed: {e}")
            ocr_text = {}

        got = 0
        for n in scanned:
            t = (ocr_text.get(n) or "").strip()
            if t:
                got += 1
                out.append(Document(page_content=t,
                                    metadata=_meta(file_path, n, session_id, ocr=True)))
            elif n in fallback:
                out.append(Document(page_content=fallback[n],
                                    metadata=_meta(file_path, n, session_id)))
        print(f"  OCR: requested {len(scanned)} page(s), got text for {got}")
    return out

def extract_txt(file_path: str, session_id: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return [Document(
        page_content=text,
        metadata={
            "source":     os.path.basename(file_path),
            "page":       1,
            "session_id": session_id
        }
    )]


def extract_docx(file_path: str, session_id: str) -> list[Document]:
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc       = DocxDocument(file_path)
    out       = []
    page_text = []
    page_num  = 1

    def flush():
        nonlocal page_text, page_num
        if page_text:
            out.append(Document(
                page_content="\n".join(page_text),
                metadata=_meta(file_path, page_num, session_id),
            ))
            page_text = []
            page_num += 1

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            t = Paragraph(child, doc).text.strip()
            if t:
                page_text.append(t)
            if len(page_text) >= 15:        # same pseudo-page rule as before
                flush()
        elif child.tag.endswith("}tbl"):
            table = Table(child, doc)
            rows = []
            for row in table.rows:
                seen, cells = set(), []
                for cell in row.cells:      # merged cells repeat: keep the first
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    cells.append(cell.text)
                rows.append(cells)
            md = rows_to_markdown(rows)
            if md:
                out.append(Document(
                    page_content=md,
                    metadata=_meta(file_path, page_num, session_id, content_type="table"),
                ))
    flush()
    return out

def extract_pptx(file_path: str, session_id: str) -> list[Document]:
    from pptx import Presentation
    prs = Presentation(file_path)
    out = []
    for i, slide in enumerate(prs.slides):
        page_no, texts, tables = i + 1, [], []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[c.text for c in r.cells] for r in shape.table.rows]
                md = rows_to_markdown(rows)
                if md:
                    tables.append(md)
            elif hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())

        if texts:
            out.append(Document(
                page_content="\n".join(texts),
                metadata=_meta(file_path, page_no, session_id),
            ))
        for md in tables:
            out.append(Document(
                page_content=md,
                metadata=_meta(file_path, page_no, session_id, content_type="table"),
            ))
    return out


def extract_document(file_path: str, session_id: str) -> list[Document]:
    """Router — picks the right extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    extractors = {
        ".pdf":  extract_pdf,
        ".txt":  extract_txt,
        ".md":   extract_txt,   # same as txt
        ".docx": extract_docx,
        ".pptx": extract_pptx,
    }
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return extractors[ext](file_path, session_id)

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(pages: list[Document]) -> list[Document]:
    text_docs  = [p for p in pages if p.metadata.get("content_type") != "table"]
    table_docs = [p for p in pages if p.metadata.get("content_type") == "table"]

    chunks = splitter.split_documents(text_docs)
    for t in table_docs:
        for part in split_table(t.page_content, t.metadata["page"]):
            chunks.append(Document(page_content=part, metadata=dict(t.metadata)))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    print(f"  created {len(chunks)} chunks ({len(table_docs)} tables)")
    return chunks

# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[Document]) -> tuple:
    """Build BM25 index in memory — no disk persistence."""
    tokenized = [re.findall(r"\w+", doc.page_content.lower()) for doc in chunks]
    bm25      = BM25Okapi(tokenized)
    return bm25, chunks

# ── Vector store ──────────────────────────────────────────────────────────────

def store_in_pgvector(chunks: list[Document]) -> PGVector:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    engine = create_engine(
        CONNECTION_STRING,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    vs = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
    )
    vs.add_documents(chunks, batch_size=len(chunks))
    return vs

# ── Main ingestion pipeline ───────────────────────────────────────────────────

def ingest_pdf(file_path: str, session_id: str) -> tuple:
    """
    Full ingestion pipeline.
    Accepts PDF, TXT, MD, DOCX, PPTX.
    Returns (vectorstore, bm25, docs)
    """
    filename = os.path.basename(file_path)
    ext      = os.path.splitext(filename)[1].lower()

    print(f"\nIngesting: {filename} [{ext}] for session {session_id[:8]}...")

    with StageTimer() as t:
        pages = extract_document(file_path, session_id)
    print(f"  extraction: {t.elapsed_ms:.0f}ms — {len(pages)} pages")

    # page limit check
    page_count = len({p.metadata["page"] for p in pages})
    if page_count > MAX_PAGES:
        raise ValueError(
            f"Document has {page_count} pages. "
            f"Maximum allowed is {MAX_PAGES}. "
            f"Please upload a shorter document."
        )

    with StageTimer() as t:
        chunks = chunk_documents(pages)
    print(f"  chunking: {t.elapsed_ms:.0f}ms")

    # token count
    enc          = tiktoken.encoding_for_model("text-embedding-ada-002")
    exact_tokens = sum(len(enc.encode(c.page_content)) for c in chunks)
    usage        = TokenUsage(embedding_tokens=exact_tokens)
    print(f"  {exact_tokens} embedding tokens — ${usage.embedding_cost:.6f}")

    with StageTimer() as t:
        vs = store_in_pgvector(chunks)
    print(f"  embed+store: {t.elapsed_ms:.0f}ms")

    with StageTimer() as t:
        bm25, docs = build_bm25_index(chunks)
    print(f"  BM25 index: {t.elapsed_ms:.0f}ms")

    print(f"\nDone. {len(chunks)} chunks ready.\n")
    return vs, bm25, docs
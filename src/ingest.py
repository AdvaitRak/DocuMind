# src/ingest.py

import os
import fitz                          # PyMuPDF
import pickle
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from rank_bm25 import BM25Okapi
import tiktoken

load_dotenv()


CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME   = "documind_chunks"
BM25_INDEX_PATH   = "indexes/bm25_index.pkl"


embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # text-embedding-ada-002
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)


def extract_pages(pdf_path: str) -> list[Document]:
    """
    Extract text page by page using PyMuPDF.
    Each page becomes one LangChain Document with metadata.
    Metadata carries source filename + page number for citations later.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append(Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page":   i + 1
                }
            ))
    doc.close()
    print(f"  extracted {len(pages)} pages")
    return pages


def chunk_documents(pages: list[Document]) -> list[Document]:
    """
    Split pages into overlapping chunks.
    splitter preserves metadata from parent Document automatically.
    chunk_index added manually for traceability.
    """
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    print(f"  created {len(chunks)} chunks")
    return chunks


def build_bm25_index(chunks: list[Document]):
    """
    Build BM25 index from all chunks and save to disk.
    Called after every ingestion so index stays in sync with vector store.
    At query time we load this once — never rebuild per query.
    """
    os.makedirs("indexes", exist_ok=True)
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25      = BM25Okapi(tokenized)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)

    print(f"  BM25 index saved → {BM25_INDEX_PATH}")


def store_in_pgvector(chunks: list[Document]) -> PGVector:
    """
    Embed chunks using Azure text-embedding-ada-002 and store in Supabase pgvector.
    LangChain handles batching, DB connection, and table creation automatically.
    """
    print(f"  embedding + storing {len(chunks)} chunks in pgvector...")
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=False    # append, don't wipe existing data
    )
    print("stored in pgvector")
    return vectorstore


def ingest_pdf(pdf_path: str):
    """
    Full ingestion pipeline:
    PDF → extract pages → chunk → embed → pgvector + BM25 index
    """
    print(f"\nIngesting: {os.path.basename(pdf_path)}")

    from src.metrics import StageTimer, cost_tracker, TokenUsage

    # 1. extract
    with StageTimer() as t:
        pages = extract_pages(pdf_path)
    print(f"  extraction: {t.elapsed_ms:.0f}ms")

    # 2. chunk
    with StageTimer() as t:
        chunks = chunk_documents(pages)
    print(f"  chunking: {t.elapsed_ms:.0f}ms")

    # 3. embed + store (timed together since Azure call is inside store)
    with StageTimer() as t:
        vectorstore = store_in_pgvector(chunks)
    print(f"  embed+store: {t.elapsed_ms:.0f}ms")

    # track embedding token usage
    # ada-002 tokenizes at ~0.75 tokens per word
    # track embedding token usage — exact count via tiktoken
    enc= tiktoken.encoding_for_model("text-embedding-ada-002")
    exact_tokens= sum(len(enc.encode(c.page_content)) for c in chunks)
    usage = TokenUsage(embedding_tokens=exact_tokens)
    print(f"  {exact_tokens} embedding tokens (exact)")
    print(f"  ${usage.embedding_cost:.6f} embedding cost")

    # 4. build BM25 index from all chunks
    with StageTimer() as t:
        build_bm25_index(chunks)
    print(f"  BM25 indexing: {t.elapsed_ms:.0f}ms")

    print(f"\nDone. {len(chunks)} chunks ready for retrieval.\n")
    return vectorstore


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <path_to_pdf>")
        sys.exit(1)
    ingest_pdf(sys.argv[1])
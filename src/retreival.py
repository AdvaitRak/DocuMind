# src/retrieval.py

import os
import pickle
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.metrics import StageTimer, PipelineMetrics

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import time
from sqlalchemy.exc import OperationalError

load_dotenv()

CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME   = "documind_chunks"
BM25_INDEX_PATH   = os.getenv("BM25_INDEX_PATH", "indexes/bm25_index.pkl")


embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    backend="onnx",
    model_kwargs={"file_name": "onnx/model_quint8_avx2.onnx"} 
)
print(f"Reranker loaded — id: {id(reranker)}")  # prints memory address

def create_engine_with_retry(url: str, retries: int = 3, delay: float = 2.0):
    for attempt in range(retries):
        try:
            eng = create_engine(
                url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=300,
            )
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"  DB connected on attempt {attempt + 1}")
            return eng
        except OperationalError as e:
            print(f"  DB connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise Exception("Failed to connect to DB after retries")


engine = create_engine_with_retry(os.getenv("DATABASE_URL"))

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=engine,
)



def load_bm25() -> tuple[BM25Okapi, list[Document]]:
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run ingest first."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"BM25 index loaded — {len(data['docs'])} docs")
    return data["bm25"], data["docs"]




def dense_search(query: str,session_id: str, top_k: int = 50) -> list[Document]:
    print(f"  dense_search session_id type: {type(session_id)} value: {session_id}")
    return vectorstore.similarity_search(query,
                                        k=top_k,
                                        filter={"session_id": session_id})



def sparse_search(query: str, session_id: str, top_k: int = 50) -> list[Document]:
    from main import sessions

    session = sessions.get(session_id)
    if not session or not session["bm25"]:
        print(f"  No BM25 index for session {session_id[:8]} — skipping sparse search")
        return []

    tokens = query.lower().split()
    scores = session["bm25"].get_scores(tokens)
    scored = sorted(
        zip(session["docs"], scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, _ in scored[:top_k]]


def rrf_fusion(
    dense_results:  list[Document],
    sparse_results: list[Document],
    k: int = 60
) -> list[Document]:
    scores = {}

    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1 / (k + rank + 1)

    for rank, doc in enumerate(sparse_results):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1 / (k + rank + 1)

    sorted_docs = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    return [item["doc"] for item in sorted_docs]



def rerank(
    query: str,
    docs:  list[Document],
    top_n: int = 5
) -> list[Document]:
    import time
    print("  rerank() entered")
    t0 = time.perf_counter()
    if not docs:
        print("  No docs to rerank — returning empty list")
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    print(f"  pairs built: {(time.perf_counter()-t0)*1000:.0f}ms")
    print(f"  avg doc length: {sum(len(d.page_content) for d in docs)//len(docs)} chars")
    
    t1 = time.perf_counter()
    scores = reranker.predict(pairs)
    print(f"  predict: {(time.perf_counter()-t1)*1000:.0f}ms")
    
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_n]]


def retrieve_dense_only(query: str, metrics: PipelineMetrics, session_id: str) -> list[Document]:
    with StageTimer() as t:
        results = dense_search(query, session_id, top_k=5)
    metrics.dense_retrieval_ms = t.elapsed_ms

    # normalize vector scores to 0-1
    if results:
        max_v = max(1 / (60 + i + 1) for i in range(len(results))) + 1e-8
        for i, doc in enumerate(results):
            raw = 1 / (60 + i + 1)
            doc.metadata["vector_score"]   = round(raw / max_v, 4)
            doc.metadata["bm25_score"]     = 0.0
            doc.metadata["reranker_score"] = 0.0

    metrics.chunks_retrieved    = len(results)
    metrics.chunks_after_rerank = len(results)
    return results


def retrieve_hybrid(query: str, metrics: PipelineMetrics, session_id: str) -> list[Document]:
    with StageTimer() as t:
        dense_hits = dense_search(query, session_id, top_k=10)
    metrics.dense_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        sparse_hits = sparse_search(query, session_id, top_k=10)
    metrics.sparse_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        fused = rrf_fusion(dense_hits, sparse_hits)
    metrics.rrf_fusion_ms = t.elapsed_ms

    # attach RRF scores
    dense_contents  = {d.page_content: i for i, d in enumerate(dense_hits)}
    sparse_contents = {d.page_content: i for i, d in enumerate(sparse_hits)}

    for doc in fused:
        doc.metadata["vector_score"] = round(1 / (60 + dense_contents.get(doc.page_content, 60) + 1), 4)
        doc.metadata["bm25_score"]   = round(1 / (60 + sparse_contents.get(doc.page_content, 60) + 1), 4)
        doc.metadata["reranker_score"] = 0.0

    # normalize vector and bm25 to 0-1
    if fused:
        max_v = max(d.metadata["vector_score"] for d in fused) + 1e-8
        max_b = max(d.metadata["bm25_score"]   for d in fused) + 1e-8
        for doc in fused:
            doc.metadata["vector_score"] = round(doc.metadata["vector_score"] / max_v, 4)
            doc.metadata["bm25_score"]   = round(doc.metadata["bm25_score"]   / max_b, 4)

    top_5 = fused[:5]
    metrics.chunks_retrieved    = len(fused)
    metrics.chunks_after_rerank = len(top_5)
    return top_5


def retrieve_hybrid_rerank(query: str, metrics: PipelineMetrics, session_id: str) -> list[Document]:
    with StageTimer() as t:
        dense_hits = dense_search(query, session_id, top_k=10)
    metrics.dense_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        sparse_hits = sparse_search(query, session_id, top_k=10)
    metrics.sparse_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        fused = rrf_fusion(dense_hits, sparse_hits)
    metrics.rrf_fusion_ms = t.elapsed_ms

    # ── attach RRF scores ─────────────────────────────────────────────────────
    dense_contents  = {d.page_content: i for i, d in enumerate(dense_hits)}
    sparse_contents = {d.page_content: i for i, d in enumerate(sparse_hits)}

    for doc in fused:
        doc.metadata["vector_score"] = round(1 / (60 + dense_contents.get(doc.page_content, 60) + 1), 4)
        doc.metadata["bm25_score"]   = round(1 / (60 + sparse_contents.get(doc.page_content, 60) + 1), 4)

    with StageTimer() as t:
        pairs           = [(query, doc.page_content) for doc in fused]
        reranker_scores = reranker.predict(pairs) if fused else []

        # ── normalize reranker scores to 0-1 ─────────────────────────────────
        if len(reranker_scores) > 0:
            min_s = float(min(reranker_scores))
            max_s = float(max(reranker_scores))
            span  = max_s - min_s + 1e-8
            normalized = [round((float(s) - min_s) / span, 4) for s in reranker_scores]
        else:
            normalized = []

        for doc, score in zip(fused, normalized):
            doc.metadata["reranker_score"] = score

        final = sorted(zip(fused, normalized), key=lambda x: x[1], reverse=True)[:5]
        final = [doc for doc, _ in final]

    metrics.rerank_ms = t.elapsed_ms

    metrics.chunks_retrieved    = len(fused)
    metrics.chunks_after_rerank = len(final)
    return final

PIPELINE_MAP = {
    "dense_only":      retrieve_dense_only,
    "hybrid":          retrieve_hybrid,
    "hybrid_rerank":   retrieve_hybrid_rerank,
}

def retrieve(
    query:    str,
    metrics:  PipelineMetrics,
    session_id: str,
    pipeline: str = "hybrid_rerank"
    
) -> list[Document]:
    """
    Call this from generation.py.
    pipeline: "dense_only" | "hybrid" | "hybrid_rerank"
    """
    if pipeline not in PIPELINE_MAP:
        raise ValueError(f"Unknown pipeline: {pipeline}. Choose from {list(PIPELINE_MAP.keys())}")

    return PIPELINE_MAP[pipeline](query, metrics,session_id)

if __name__ == "__main__":
    from src.metrics import PipelineMetrics

    query = "what are your technical skills"  # change this to anything
    session_id = "Enter yours session ID"
    metrics = PipelineMetrics(
        pipeline_variant="hybrid_rerank",
        query=query
    )

    results = retrieve(query, metrics,session_id, pipeline="hybrid_rerank")

    print(f"\n── Retrieved {len(results)} chunks ──────────────────────")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"    {doc.page_content[:200]}...")
    
    print("\n── Latency ──────────────────────────────────────────")
    print(f"  Dense:   {metrics.dense_retrieval_ms:.0f}ms")
    print(f"  Sparse:  {metrics.sparse_retrieval_ms:.0f}ms")
    print(f"  RRF:     {metrics.rrf_fusion_ms:.0f}ms")
    print(f"  Rerank:  {metrics.rerank_ms:.0f}ms")
    print(f"  Total:   {metrics.total_latency_ms:.0f}ms")
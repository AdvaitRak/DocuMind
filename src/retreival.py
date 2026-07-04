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

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


def load_bm25() -> tuple[BM25Okapi, list[Document]]:
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run ingest first."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"BM25 index loaded — {len(data['docs'])} docs")
    return data["bm25"], data["docs"]

bm25_index, bm25_corpus = load_bm25()



def dense_search(query: str, top_k: int = 50) -> list[Document]:
    return vectorstore.similarity_search(query, k=top_k)



def sparse_search(query: str, top_k: int = 50) -> list[Document]:
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    scored = sorted(
        zip(bm25_corpus, scores),
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
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, _ in reranked[:top_n]]



def retrieve_dense_only(
    query:   str,
    metrics: PipelineMetrics
) -> list[Document]:
    """Pipeline A — baseline"""
    with StageTimer() as t:
        results = dense_search(query, top_k=5)
    metrics.dense_retrieval_ms  = t.elapsed_ms
    metrics.chunks_retrieved    = len(results)
    metrics.chunks_after_rerank = len(results)
    return results


def retrieve_hybrid(
    query:   str,
    metrics: PipelineMetrics
) -> list[Document]:
    """Pipeline B — dense + sparse + RRF, no reranking"""
    with StageTimer() as t:
        dense_hits = dense_search(query, top_k=50)
    metrics.dense_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        sparse_hits = sparse_search(query, top_k=50)
    metrics.sparse_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        fused = rrf_fusion(dense_hits, sparse_hits)
    metrics.rrf_fusion_ms = t.elapsed_ms

    top_5 = fused[:5]
    metrics.chunks_retrieved    = len(fused)
    metrics.chunks_after_rerank = len(top_5)
    return top_5


def retrieve_hybrid_rerank(
    query:   str,
    metrics: PipelineMetrics
) -> list[Document]:
    """Pipeline C — dense + sparse + RRF + cross-encoder rerank"""
    with StageTimer() as t:
        dense_hits = dense_search(query, top_k=50)
    metrics.dense_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        sparse_hits = sparse_search(query, top_k=50)
    metrics.sparse_retrieval_ms = t.elapsed_ms

    with StageTimer() as t:
        fused = rrf_fusion(dense_hits, sparse_hits)
    metrics.rrf_fusion_ms = t.elapsed_ms

    with StageTimer() as t:
        final = rerank(query, fused, top_n=5)
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
    pipeline: str = "hybrid_rerank"
) -> list[Document]:
    """
    Call this from generation.py.
    pipeline: "dense_only" | "hybrid" | "hybrid_rerank"
    """
    if pipeline not in PIPELINE_MAP:
        raise ValueError(f"Unknown pipeline: {pipeline}. Choose from {list(PIPELINE_MAP.keys())}")

    return PIPELINE_MAP[pipeline](query, metrics)

if __name__ == "__main__":
    from src.metrics import PipelineMetrics

    query = "what are your technical skills"  # change this to anything

    metrics = PipelineMetrics(
        pipeline_variant="hybrid_rerank",
        query=query
    )

    results = retrieve(query, metrics, pipeline="hybrid_rerank")

    print(f"\n── Retrieved {len(results)} chunks ──────────────────────")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"    {doc.page_content[:200]}...")
    
    print(f"\n── Latency ──────────────────────────────────────────")
    print(f"  Dense:   {metrics.dense_retrieval_ms:.0f}ms")
    print(f"  Sparse:  {metrics.sparse_retrieval_ms:.0f}ms")
    print(f"  RRF:     {metrics.rrf_fusion_ms:.0f}ms")
    print(f"  Rerank:  {metrics.rerank_ms:.0f}ms")
    print(f"  Total:   {metrics.total_latency_ms:.0f}ms")
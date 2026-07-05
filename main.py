# main.py

import os
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

load_dotenv()

print("\nLoading models...")
from src.retreival import reranker, bm25_index, vectorstore
from src.generation import llm
from src.metrics import cost_tracker
print("Models ready.\n")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n── DocuMind starting up ─────────────────────────────")
    
    
    print("  ✓ Reranker loaded")
    print("  ✓ BM25 index loaded")
    print("  ✓ DB connection pool ready")
    print("  ✓ LLM ready")
    print("\n── DocuMind ready ───────────────────────────────────\n")
    
    yield  # server runs here
    
    # shutdown
    print("\n── DocuMind shutting down ───────────────────────────")
    cost_tracker.report()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind",
    description="Production RAG pipeline with hybrid search, reranking, and observability",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str
    pipeline:  Optional[str] = "hybrid_rerank"  # dense_only | hybrid | hybrid_rerank

class Source(BaseModel):
    source: str
    page:   int
    chunk:  str

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[Source]
    metrics:  dict
    run_id:   str

class IngestRequest(BaseModel):
    pdf_path: str

class HealthResponse(BaseModel):
    status:  str
    version: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main RAG endpoint.
    Accepts a question, runs the full pipeline, returns answer + sources + metrics.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.pipeline not in ["dense_only", "hybrid", "hybrid_rerank"]:
        raise HTTPException(
            status_code=400,
            detail="pipeline must be one of: dense_only, hybrid, hybrid_rerank"
        )

    run_id = str(uuid.uuid4())

    try:
        from src.generation import generate
        result = generate(
            question=request.question,
            pipeline=request.pipeline,
            run_id=run_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        metrics=result["metrics"],
        run_id=run_id
    )


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Ingest a PDF into the vector store.
    Rebuilds BM25 index after ingestion.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.pdf_path}"
        )

    try:
        from src.ingest import ingest_pdf
        ingest_pdf(request.pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", "file": request.pdf_path}


@app.get("/cost")
async def cost_report():
    """Return cumulative cost and token usage across all queries this session."""
    from src.metrics import cost_tracker
    return {
        "total_queries":  cost_tracker.total_queries,
        "total_tokens":   cost_tracker.total_tokens,
        "total_cost_usd": round(cost_tracker.total_cost_usd, 6),
        "by_pipeline":    {
            k: round(v, 6)
            for k, v in cost_tracker.by_pipeline.items()
        }
    }


@app.post("/compare")
async def compare(request: QueryRequest):
    """
    Run the same question through all three pipeline variants.
    Returns side by side metrics for A/B comparison.
    """
    from src.generation import compare_pipelines
    results = compare_pipelines(request.question)

    # strip full answer, just return metrics + pipeline name
    comparison = {}
    for pipeline, result in results.items():
        comparison[pipeline] = {
            "answer":  result["answer"],
            "metrics": result["metrics"]
        }

    return comparison
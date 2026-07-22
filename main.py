# main.py

import os
import uuid
import json
import shutil
import aiofiles
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text

load_dotenv()
limiter = Limiter(key_func=get_remote_address)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}
# ── Load models at startup ────────────────────────────────────────────────────
print("\nLoading models...")
from src.retreival import vectorstore, engine
from src.generation import llm
from src.metrics import cost_tracker
from src.ingest import ingest_pdf
print("Models ready.\n")

# ── Session store ─────────────────────────────────────────────────────────────
sessions: dict = {}
SESSION_TIMEOUT_MINUTES = 30
scheduler = AsyncIOScheduler()

# ── Session helpers ───────────────────────────────────────────────────────────

async def delete_session(session_id: str):
    """Delete all chunks for a session from pgvector and RAM."""
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "DELETE FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'session_id' = :sid"
                ),
                {"sid": session_id}
            )
            conn.commit()
        print(f"  pgvector chunks deleted for session: {session_id[:8]}")
    except Exception as e:
        print(f"  Session delete error: {e}")

    sessions.pop(session_id, None)
    print(f"  Session cleared from RAM: {session_id[:8]}")


async def cleanup_expired_sessions():
    """Runs every 5 minutes — deletes inactive sessions."""
    now = datetime.now(timezone.utc)
    expired = [
        sid for sid, data in sessions.items()
        if now - data["last_active"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    ]
    for sid in expired:
        print(f"  Auto-expiring session: {sid[:8]}")
        await delete_session(sid)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("── DocuMind ready ───────────────────────────────────\n")
    scheduler.add_job(cleanup_expired_sessions, "interval", minutes=5)
    scheduler.start()
    yield
    scheduler.shutdown()
    print("\n── DocuMind shutting down ───────────────────────────")
    cost_tracker.report()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind",
    description="Production RAG pipeline with hybrid search, reranking, and observability",
    version="1.0.0",
    lifespan=lifespan
)

# add these two lines
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    pipeline:   Optional[str] = "hybrid_rerank"
    session_id: str

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

class HealthResponse(BaseModel):
    status:  str
    version: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/session/start")
@limiter.limit("10/minute") 
async def start_session(request: Request):
    """Create a new session. Call this when user opens the app."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "bm25":        None,
        "docs":        [],
        "last_active": datetime.now(timezone.utc)
    }
    print(f"  Session started: {session_id[:8]}")
    return {"session_id": session_id}


@app.post("/session/end")
async def end_session(request: Request,session_id: str):
    """Delete session data. Call this when user closes the app."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    await delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.post("/ingest")
@limiter.limit("3/minute")
async def ingest(
    request:    Request,
    session_id: str,
    file:       UploadFile = File(...)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, TXT, MD, DOCX, PPTX"
        )

    os.makedirs("data/pdfs", exist_ok=True)
    temp_path = f"data/pdfs/{file.filename}"

    # ── async file write ──────────────────────────────────────────────────────
    async with aiofiles.open(temp_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    try:
        vs, bm25, docs = ingest_pdf(temp_path, session_id)
        sessions[session_id]["bm25"]        = bm25
        sessions[session_id]["docs"]        = docs
        sessions[session_id]["last_active"] = datetime.now(timezone.utc)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status":  "success",
        "file":    file.filename,
        "session": session_id
    }

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, query_request: QueryRequest):
    if not query_request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if query_request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if query_request.pipeline not in ["dense_only", "hybrid", "hybrid_rerank"]:
        raise HTTPException(status_code=400, detail="Invalid pipeline")

    sessions[query_request.session_id]["last_active"] = datetime.now(timezone.utc)
    run_id = str(uuid.uuid4())

    try:
        from src.generation import generate
        result = generate(
            question=query_request.question,
            pipeline=query_request.pipeline,
            session_id=query_request.session_id,
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


@app.post("/query/stream")
@limiter.limit("10/minute")
async def query_stream(request: Request, query_request: QueryRequest):
    if not query_request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if query_request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if query_request.pipeline not in ["dense_only", "hybrid", "hybrid_rerank"]:
        raise HTTPException(status_code=400, detail="Invalid pipeline")

    sessions[query_request.session_id]["last_active"] = datetime.now(timezone.utc)

    async def event_generator():
        try:
            from src.generation import generate_stream
            async for event in generate_stream(
                question=query_request.question,
                pipeline=query_request.pipeline,
                session_id=query_request.session_id,
            ):
                # debug — print event type before serializing
                print(f"  event type: {event.get('type')} — {type(event)}")
                serialized = json.dumps(event)   # ← will raise here if not serializable
                yield f"data: {serialized}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()               # ← prints full stack trace to terminal
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/cost")
@limiter.limit("10/minute")
async def cost_report(request: Request):
    from src.metrics import cost_tracker
    return {
        "total_queries":  cost_tracker.total_queries,
        "total_tokens":   cost_tracker.total_tokens,
        "total_cost_usd": round(cost_tracker.total_cost_usd, 6),
        "by_pipeline":    {k: round(v, 6) for k, v in cost_tracker.by_pipeline.items()}
    }


@app.post("/compare")
@limiter.limit("2/minute") 
async def compare(request: Request, query_request: QueryRequest):
    if query_request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    from src.generation import compare_pipelines
    results = compare_pipelines(query_request.question, query_request.session_id)

    return {
        pipeline: {
            "answer":  result["answer"],
            "metrics": result["metrics"]
        }
        for pipeline, result in results.items()
    }
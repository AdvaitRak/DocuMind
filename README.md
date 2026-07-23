# DocuMind — Production RAG Backend

> A production-grade document intelligence API with hybrid retrieval, cross-encoder reranking, SSE streaming, and full observability. Built to go beyond basic vector search.

---

## Live Demo

**Backend API:** `https://documind-backend-231b825b5d70.herokuapp.com`  

**Frontend:** `https://docu-mind.tech`

**API Docs:** `https://documind-backend-231b825b5d70.herokuapp.com/docs`

---

## Architecture

```
User uploads PDF / DOCX / PPTX / TXT / MD
          ↓
    PyMuPDF / python-docx / python-pptx extractor
          ↓
    RecursiveCharacterTextSplitter (512 tokens, 100 overlap)
          ↓
    Azure OpenAI text-embedding-ada-002 (1536-dim)
          ↓
    pgvector (Supabase Mumbai) + BM25 index (session RAM)
          ↓
─────────────── Query Pipeline ───────────────
          ↓
    Dense search (pgvector cosine similarity)
          +
    Sparse search (BM25, ~0ms, in-memory)
          ↓
    RRF Fusion  score = 1/(60+dense_rank) + 1/(60+sparse_rank)
          ↓
    Cohere Rerank v3.5 (cross-encoder, API)
          ↓
    Top 5 chunks → Gemini / Llama via OpenRouter
          ↓
    SSE streaming response with citations
          ↓
    LangSmith trace (latency + tokens + cost + RAGAS)
```

---

## Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + LangChain 0.3+ |
| Embeddings | Azure OpenAI `text-embedding-ada-002` (UAE North) |
| Vector DB | pgvector on Supabase (Mumbai, ap-south-1) |
| Sparse search | BM25 via `rank-bm25` (in-memory, per-session) |
| Reranking | Cohere Rerank v3.5 API |
| LLM | OpenRouter (Llama 3.3 70B / Gemini 2.5 Flash) |
| Streaming | Server-Sent Events (SSE) via FastAPI `StreamingResponse` |
| Observability | LangSmith (APAC endpoint) |
| Evaluation | RAGAS 0.4.3 |
| Session management | UUID sessions + APScheduler cleanup |
| Rate limiting | SlowAPI per-IP |
| Deployment | Heroku Basic dyno (Docker container) |

---

## Retrieval Pipeline — Three A/B Variants

Every query can be routed through one of three pipeline variants, each logging full metrics to LangSmith for comparison:

**Pipeline A — Dense only (baseline)**
```
query → embed → pgvector similarity search → top 5 → LLM
```

**Pipeline B — Hybrid**
```
query → dense search (top 20) + BM25 sparse search (top 20)
      → RRF fusion → top 5 → LLM
```

**Pipeline C — Hybrid + Rerank (production default)**
```
query → dense search (top 10) + BM25 sparse search (top 10)
      → RRF fusion → Cohere Rerank v3.5 → top 5 → LLM
```

---

## RAGAS Evaluation Results

Evaluated on *Attention Is All You Need* (Vaswani et al., 2017) using `hybrid_rerank` pipeline:

| Metric | Score | What it means |
|---|---|---|
| **Faithfulness** | **0.962** | 96.2% of answer claims supported by retrieved chunks |
| **Answer relevancy** | **0.966** | Answers directly address the question 96.6% of the time |
| **Context precision** | **0.767** | 76.7% of retrieved chunks were genuinely useful |

---

## Observability — LangSmith Dashboard

Every query is traced end-to-end in LangSmith with per-stage latency, token counts, dollar cost, and RAGAS scores attached as feedback.

### Trace Count (Last 7 Days)
Peak of 21 traces on 7/22 — each spike corresponds to a testing session. Success rate near 100% across all traces.

### Trace Latency Percentiles
P50 latency settling at **~2s** after warm startup. Early spikes (6-8s) correspond to cold-start model loading and Supabase connection establishment — resolved after first request.

### Trace Error Rate
Error rate <5% — spikes correspond to OpenRouter transient JSON injection errors in SSE stream, handled by retry logic.

---

## Performance Results

### Latency Breakdown (warm, hybrid_rerank, production)

| Stage | Latency |
|---|---|
| Dense retrieval (Azure UAE → Supabase Mumbai) | ~800ms |
| Sparse retrieval (BM25, in-memory) | ~0ms |
| RRF fusion (in-memory) | ~0ms |
| Reranking (Cohere API) | ~200ms |
| LLM generation (OpenRouter, streaming) | ~800ms |
| **Total end-to-end** | **~1,800ms** |

### Optimisations and impact

| Optimisation | Before | After | Improvement |
|---|---|---|---|
| Supabase region (Seoul → Mumbai) | 2,500ms | 800ms | **68% reduction** |
| SQLAlchemy connection pooling | 2,500ms cold | 78ms warm | **97% reduction** |
| BM25 pre-built at ingestion | O(n) per query | ~0ms | **eliminated per-query rebuild** |
| Cohere API reranking (vs local torch) | 500MB slug | ~180MB slug | **64% slug size reduction** |
| ONNX quantized reranker (local dev) | 91MB model | 23MB model | **75% model size reduction** |
| ONNX inference (local dev) | ~400ms | ~150ms | **62% faster reranking** |

### Cost per query

| Component | Cost |
|---|---|
| Embeddings (ada-002, ~5 tokens) | ~$0.000001 |
| Reranking (Cohere v3.5) | $0.001 |
| LLM generation (~700 input + ~150 output tokens) | ~$0.00015 |
| **Total per query** | **~$0.00115** |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/session/start` | Create isolated session — returns `session_id` |
| `POST` | `/session/end` | Delete session + all chunks from pgvector |
| `POST` | `/ingest` | Upload PDF/DOCX/PPTX/TXT/MD (max 20 pages) |
| `POST` | `/query` | RAG query — returns full answer + sources + metrics |
| `POST` | `/query/stream` | SSE streaming RAG query |
| `POST` | `/compare` | Run query through all 3 pipeline variants |
| `GET` | `/cost` | Cumulative cost + token report for session |

### Request format — `/query/stream`

```json
{
  "question": "what is the attention mechanism?",
  "pipeline": "hybrid_rerank",
  "session_id": "f690b8b2-4c21-4a68-ab63-69cb8b6ceae1"
}
```

### SSE response format

```
data: {"type": "token", "token": "The attention mechanism..."}
data: {"type": "token", "token": " allows the model to..."}
...
data: {"type": "done", "sources": [...], "metrics": {...}}
```

---

## Session Architecture

Each user gets a UUID session. Documents are fully isolated per session:

```
POST /session/start  →  session_id: "abc123"
POST /ingest?session_id=abc123 + PDF
POST /query/stream   { session_id: "abc123", question: "..." }
POST /session/end?session_id=abc123  →  chunks deleted from pgvector
```

Auto-cleanup: APScheduler runs every 5 minutes, deletes sessions inactive >30 minutes. Server shutdown cleans all active sessions.

pgvector deletion: `DELETE FROM langchain_pg_embedding WHERE cmetadata->>'session_id' = :sid`

---

## Multi-format Document Support

| Format | Library | Notes |
|---|---|---|
| `.pdf` | PyMuPDF | Page-by-page extraction with metadata |
| `.docx` | python-docx | Paragraph grouping into virtual pages |
| `.pptx` | python-pptx | One Document per slide |
| `.txt` / `.md` | built-in Python | Full text as single Document |

Page limit: 20 pages max per upload (configurable via `MAX_PDF_PAGES` env var). Returns 400 with clear message if exceeded.

---

## Local Development

```bash
# clone
git clone https://github.com/yourusername/documind-backend
cd documind-backend

# install dependencies
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# copy env
cp .env.example .env
# fill in your keys

# run
uvicorn main:app --port 8000

# test
curl http://localhost:8000/health
```

---

## Docker

```bash
# build
docker build -t documind-backend .

# run
docker run -p 8000:8000 --env-file .env documind-backend
```

---

## Environment Variables

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Database (Supabase pgvector)
DATABASE_URL=postgresql+psycopg://...

# LLM
OPENROUTER_API_KEY=

# Reranking
COHERE_API_KEY=

# Observability
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=Documind
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=Documind

# Config
MAX_PDF_PAGES=20
SESSION_TIMEOUT_MINUTES=30
```

---

## Deployment

Deployed on **Heroku Basic dyno** (Docker container stack) with **Supabase Mumbai** for pgvector.

```bash
heroku stack:set container -a documind-backend
heroku config:set KEY=value -a documind-backend
git push heroku main
```

---

## What makes this different from a basic RAG

Most RAG tutorials stop at "embed + similarity search + generate." This pipeline adds:

- **Hybrid retrieval** — BM25 catches exact keyword matches that dense vectors miss
- **RRF fusion** — rank-based score fusion rewards consensus across retrieval systems
- **Cross-encoder reranking** — query-document token-level interaction via Cohere API
- **Session isolation** — multi-user safe, full cleanup on session end
- **RAGAS evaluation** — quantified quality metrics, not just "it works"
- **LangSmith observability** — every query traced with latency, cost, and quality scores
- **SSE streaming** — progressive response rendering, not waiting for full answer
- **A/B pipeline comparison** — measurable tradeoffs between quality and latency

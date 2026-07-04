# src/metrics.py

import time
import os
from dataclasses import dataclass, field
from typing import Optional
from langsmith import Client

# ── Azure OpenAI pricing (per token) ────────────────────────────────────────
PRICING = {
    "text-embedding-3-large": 0.00013 / 1000,
    "gpt-4o-mini": {
        "input":  0.00015 / 1000,
        "output": 0.00060 / 1000,
    }
}

langsmith_client = Client()

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StageTimer:
    """
    Tracks latency for one stage of the pipeline.
    Usage:
        with StageTimer() as t:
            do_something()
        print(t.elapsed_ms)
    """
    elapsed_ms: float = 0.0
    _start: float = field(default=0.0, repr=False)

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


@dataclass
class TokenUsage:
    embedding_tokens: int  = 0
    llm_input_tokens: int  = 0
    llm_output_tokens: int = 0

    @property
    def embedding_cost(self) -> float:
        return self.embedding_tokens * PRICING["text-embedding-3-large"]

    @property
    def llm_cost(self) -> float:
        return (
            self.llm_input_tokens  * PRICING["gpt-4o-mini"]["input"] +
            self.llm_output_tokens * PRICING["gpt-4o-mini"]["output"]
        )

    @property
    def total_cost(self) -> float:
        return self.embedding_cost + self.llm_cost

    @property
    def total_tokens(self) -> int:
        return self.embedding_tokens + self.llm_input_tokens + self.llm_output_tokens


@dataclass
class PipelineMetrics:
    """
    Single object that accumulates every metric for one query run.
    Gets logged to LangSmith at the end.
    """
    pipeline_variant: str          # "dense_only" | "hybrid" | "hybrid_rerank"
    query: str

    # latency per stage (ms)
    dense_retrieval_ms:  float = 0.0
    sparse_retrieval_ms: float = 0.0
    rrf_fusion_ms:       float = 0.0
    rerank_ms:           float = 0.0
    generation_ms:       float = 0.0

    # token + cost
    tokens: TokenUsage = field(default_factory=TokenUsage)

    # RAGAS scores — filled in by eval.py
    faithfulness:       Optional[float] = None
    answer_relevancy:   Optional[float] = None
    context_precision:  Optional[float] = None
    context_recall:     Optional[float] = None

    # retrieval quality
    chunks_retrieved:   int   = 0
    chunks_after_rerank: int  = 0

    @property
    def total_latency_ms(self) -> float:
        return (
            self.dense_retrieval_ms +
            self.sparse_retrieval_ms +
            self.rrf_fusion_ms +
            self.rerank_ms +
            self.generation_ms
        )

    def summary(self) -> dict:
        return {
            "pipeline":             self.pipeline_variant,
            "query":                self.query,
            "latency": {
                "dense_retrieval_ms":  round(self.dense_retrieval_ms,  2),
                "sparse_retrieval_ms": round(self.sparse_retrieval_ms, 2),
                "rrf_fusion_ms":       round(self.rrf_fusion_ms,       2),
                "rerank_ms":           round(self.rerank_ms,           2),
                "generation_ms":       round(self.generation_ms,       2),
                "total_ms":            round(self.total_latency_ms,    2),
            },
            "tokens": {
                "embedding":  self.tokens.embedding_tokens,
                "llm_input":  self.tokens.llm_input_tokens,
                "llm_output": self.tokens.llm_output_tokens,
                "total":      self.tokens.total_tokens,
            },
            "cost": {
                "embedding_usd": round(self.tokens.embedding_cost, 6),
                "llm_usd":       round(self.tokens.llm_cost,       6),
                "total_usd":     round(self.tokens.total_cost,     6),
            },
            "retrieval": {
                "chunks_retrieved":    self.chunks_retrieved,
                "chunks_after_rerank": self.chunks_after_rerank,
            },
            "ragas": {
                "faithfulness":      self.faithfulness,
                "answer_relevancy":  self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall":    self.context_recall,
            }
        }

    def log_to_langsmith(self, run_id: str):
        """
        Attach all metrics to a LangSmith trace as feedback scores.
        run_id comes from the LangChain callback during generation.
        """
        scores = {
            # quality
            "faithfulness":       self.faithfulness,
            "answer_relevancy":   self.answer_relevancy,
            "context_precision":  self.context_precision,
            "context_recall":     self.context_recall,
            # latency
            "total_latency_ms":   self.total_latency_ms,
            "rerank_ms":          self.rerank_ms,
            "generation_ms":      self.generation_ms,
            # cost
            "total_cost_usd":     self.tokens.total_cost,
            "total_tokens":       self.tokens.total_tokens,
        }

        for key, value in scores.items():
            if value is not None:
                langsmith_client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=value,
                    source_info={"pipeline": self.pipeline_variant}
                )

        print(f"\n📊 Metrics logged to LangSmith — run_id: {run_id}")


# ── Cumulative cost tracker ───────────────────────────────────────────────────
# Simple in-memory accumulator — resets when the process restarts.
# For persistence, write to a DB or file.

class CostTracker:
    def __init__(self):
        self.total_queries    = 0
        self.total_cost_usd   = 0.0
        self.total_tokens     = 0
        self.by_pipeline: dict[str, float] = {}

    def record(self, metrics: PipelineMetrics):
        self.total_queries  += 1
        self.total_cost_usd += metrics.tokens.total_cost
        self.total_tokens   += metrics.tokens.total_tokens

        v = metrics.pipeline_variant
        self.by_pipeline[v] = self.by_pipeline.get(v, 0.0) + metrics.tokens.total_cost

    def report(self):
        print("\n── Cumulative Cost Report ──────────────────────────")
        print(f"  Total queries : {self.total_queries}")
        print(f"  Total tokens  : {self.total_tokens:,}")
        print(f"  Total cost    : ${self.total_cost_usd:.4f}")
        print(f"  By pipeline   :")
        for pipeline, cost in self.by_pipeline.items():
            print(f"    {pipeline:20s} ${cost:.4f}")
        print("────────────────────────────────────────────────────\n")


# singleton — import this everywhere
cost_tracker = CostTracker()
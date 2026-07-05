# src/eval.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langsmith import Client
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.metrics import PipelineMetrics
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

langsmith_client = Client()

# ── RAGAS LLM + embeddings ────────────────────────────────────────────────────

ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
))

ragas_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
))

METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
]

# ── Core eval ─────────────────────────────────────────────────────────────────

def run_eval(
    question:     str,
    answer:       str,
    docs:         list[Document],
    ground_truth: str,
    metrics:      PipelineMetrics,
    run_id:       str = None,
) -> dict:
    contexts = [doc.page_content for doc in docs]
    scores   = _evaluate(question, answer, contexts, ground_truth)

    _fill_metrics(metrics, scores)
    _log_to_langsmith(run_id, metrics.pipeline_variant, scores)
    _print_scores(scores)

    return scores


# ── helpers ───────────────────────────────────────────────────────────────────

COL_MAP = {
    "faithfulness":      ["faithfulness"],
    "answer_relevancy":  ["answer_relevancy"],
    "context_precision": ["llm_context_precision_with_reference"],
    "context_recall":    ["context_recall"],
}

def _evaluate(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    sample  = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )
    dataset = EvaluationDataset(samples=[sample])

    try:
        result = evaluate(
            dataset=dataset,
            metrics=METRICS,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
            timeout=120,
        )
        scores_df = result.to_pandas()
        print(f"  Available columns: {list(scores_df.columns)}")
        return _extract_scores(scores_df)
    except Exception as e:
        print(f"  RAGAS eval error: {e}")
        return {k: None for k in COL_MAP}


def _extract_scores(scores_df) -> dict:
    scores = {}
    for our_key, possible_cols in COL_MAP.items():
        matched = next((c for c in possible_cols if c in scores_df.columns), None)
        scores[our_key] = float(scores_df[matched].iloc[0]) if matched else None
    return scores


def _fill_metrics(metrics: PipelineMetrics, scores: dict):
    metrics.faithfulness      = scores.get("faithfulness")
    metrics.answer_relevancy  = scores.get("answer_relevancy")
    metrics.context_precision = scores.get("context_precision")
    metrics.context_recall    = scores.get("context_recall")


def _log_to_langsmith(run_id: str, pipeline: str, scores: dict):
    if not run_id:
        return
    for key, value in scores.items():
        if value is None:
            continue
        try:
            langsmith_client.create_feedback(
                run_id=run_id,
                key=key,
                score=float(value),
                source_info={"pipeline": pipeline}
            )
        except Exception as e:
            print(f"  LangSmith feedback error: {e}")


def _print_scores(scores: dict):
    print(f"  Faithfulness      : {scores.get('faithfulness')}")
    print(f"  Answer relevancy  : {scores.get('answer_relevancy')}")
    print(f"  Context precision : {scores.get('context_precision')}")
    print(f"  Context recall    : {scores.get('context_recall')}")
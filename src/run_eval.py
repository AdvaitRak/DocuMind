# run_eval.py

import json
import os
from dotenv import load_dotenv
from data.test_set import test_set
from src.generation import generate
from src.eval import run_eval
from src.metrics import PipelineMetrics

load_dotenv()

PIPELINES = ["dense_only", "hybrid", "hybrid_rerank"]

def main():
    results = {p: [] for p in PIPELINES}

    for i, item in enumerate(test_set):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n{'='*60}")
        print(f"Q{i+1}: {question}")
        print(f"{'='*60}")

        for pipeline in PIPELINES:
            print(f"\n  Pipeline: {pipeline}")

            result  = generate(question, pipeline=pipeline)
            answer  = result["answer"]
            docs    = result["docs"]

            metrics = PipelineMetrics(
                pipeline_variant=pipeline,
                query=question
            )

            scores = run_eval(
                question=question,
                answer=answer,
                docs=docs,
                ground_truth=ground_truth,
                metrics=metrics,
            )

            results[pipeline].append({
                "scores":     scores,
                "latency_ms": result["metrics"]["latency"]["total_ms"],
                "cost_usd":   result["metrics"]["cost"]["total_usd"],
            })

    # aggregate + print table
    print(f"\n\n{'='*70}")
    print(f"{'Pipeline':<20} {'Faith':>8} {'Relevancy':>10} {'Precision':>10} {'Recall':>8} {'Latency':>10}")
    print(f"{'-'*70}")

    summary = {}
    for pipeline in PIPELINES:
        pr = results[pipeline]

        def avg(key):
            vals = [r["scores"].get(key) for r in pr if r["scores"].get(key) is not None]
            return sum(vals)/len(vals) if vals else 0.0

        faith     = avg("faithfulness")
        relevancy = avg("answer_relevancy")
        precision = avg("context_precision")
        recall    = avg("context_recall")
        latency   = sum(r["latency_ms"] for r in pr)/len(pr)

        summary[pipeline] = {
            "faithfulness":      round(faith, 3),
            "answer_relevancy":  round(relevancy, 3),
            "context_precision": round(precision, 3),
            "context_recall":    round(recall, 3),
            "avg_latency_ms":    round(latency, 0),
        }

        print(
            f"{pipeline:<20}"
            f"{faith:>8.3f}"
            f"{relevancy:>10.3f}"
            f"{precision:>10.3f}"
            f"{recall:>8.3f}"
            f"{latency:>9.0f}ms"
        )

    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to eval_results.json")

if __name__ == "__main__":
    main()
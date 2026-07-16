# run_eval.py — simplified

from data.test_set import test_set
from src.generation import generate
from src.eval import run_eval
from src.metrics import PipelineMetrics
import asyncio, sys, json

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    results = []
    for item in test_set[:3]:  # just 3 questions
        result  = generate(item["question"], pipeline="hybrid_rerank")
        metrics = PipelineMetrics(pipeline_variant="hybrid_rerank", query=item["question"])
        scores  = run_eval(
            question=item["question"],
            answer=result["answer"],
            docs=result["docs"],
            ground_truth=item["ground_truth"],
            metrics=metrics,
        )
        results.append(scores)

    # average scores
    avg = {
        k: round(sum(r[k] for r in results if r.get(k)) / len(results), 3)
        for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    }

    print(f"\n── hybrid_rerank RAGAS scores ───────────────────────")
    for k, v in avg.items():
        print(f"  {k:<20} : {v}")

    with open("eval_results.json", "w") as f:
        json.dump({"hybrid_rerank": avg}, f, indent=2)

if __name__ == "__main__":
    main()
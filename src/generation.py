# src/generation.py

import os
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langsmith import Client
from src.retreival import retrieve
from src.metrics import PipelineMetrics, TokenUsage, cost_tracker
#from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

langsmith_client = Client()



# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0,
#     max_tokens=1024,
#     streaming=True
# )
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    streaming=True, 
)

# cl100k_base is close enough for Llama token counting
enc = tiktoken.get_encoding("cl100k_base")
prompt_template = ChatPromptTemplate.from_template("""
You are DocuMind, a helpful AI research assistant.

First, determine the type of question:

1. If it is general conversation (greetings like "hello", "hi", "thanks", small talk, or questions clearly unrelated to any document) — respond naturally and friendly. Do not reference the context or say you lack information.

2. If it is a question about document content — answer using ONLY the context provided below and follow these rules:
   - If the answer is not in the context, say "I don't find that information in the uploaded document."
   - Be concise but complete
   - Never hallucinate facts not present in the context
   - Format your answer exactly like this example:

   **Languages:** Python, SQL, Java, C++
   *[Resume.pdf, p.1]*

   **LLM Frameworks:** LangChain, LangGraph, MCP
   *[Resume.pdf, p.1]*

   Each point on its own line, citation on the next line in italics.

Context:
{context}

Question: {question}

Answer:
""")


def format_context(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Source + page metadata included so LLM can cite correctly.
    """
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[{i+1}] (source: {source}, page {page})\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def count_tokens(text: str) -> int:
    return len(enc.encode(text))



def generate(
    question: str,
    session_id: str,
    pipeline: str = "hybrid_rerank",
    run_id:   str = None
) -> dict:
    """
    Full RAG pipeline — retrieve + generate + track metrics.
    
    Args:
        question: user query
        pipeline: "dense_only" | "hybrid" | "hybrid_rerank"
        run_id:   LangSmith run ID for logging feedback scores
    
    Returns:
        dict with answer, sources, metrics summary
    """
    from src.metrics import StageTimer

    metrics = PipelineMetrics(
        pipeline_variant=pipeline,
        query=question,
        session_id=session_id,
    )

   
    docs    = retrieve(question, metrics,session_id, pipeline=pipeline)
    context = format_context(docs)


    prompt  = prompt_template.format_messages(
        context=context,
        question=question
    )

    # count input tokens before LLM call
    input_text        = context + question
    input_token_count = count_tokens(input_text)

    with StageTimer() as t:
        response = llm.invoke(prompt)
    metrics.generation_ms = t.elapsed_ms

    answer             = response.content
    output_token_count = count_tokens(answer)

   
    metrics.tokens.llm_input_tokens  = input_token_count
    metrics.tokens.llm_output_tokens = output_token_count

   
    if run_id:
        metrics.log_to_langsmith(run_id)

   
    cost_tracker.record(metrics)

   
    sources = [
    {
        "source":         d.metadata.get("source", "unknown"),
        "page":           d.metadata.get("page", "?"),
        "chunk":          d.page_content[:100] + "...",
        "vector_score":   d.metadata.get("vector_score", 0.0),
        "bm25_score":     d.metadata.get("bm25_score", 0.0),
        "reranker_score": d.metadata.get("reranker_score", 0.0),
    }
    for d in docs
    ]

    summary = metrics.summary()
    print("\n── Generation complete ──────────────────────────────")
    print(f"  Pipeline     : {pipeline}")
    print(f"  Total latency: {summary['latency']['total_ms']:.0f}ms")
    print(f"  Input tokens : {input_token_count}")
    print(f"  Output tokens: {output_token_count}")
    print(f"  LLM cost     : ${metrics.tokens.llm_cost:.6f}")
    print(f"  Total cost   : ${metrics.tokens.total_cost:.6f}")

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "docs":     docs,
        "metrics":  metrics.summary(),
    }

# generation.py — add after generate()
async def generate_stream(
    question: str,
    session_id: str,
    pipeline: str = "hybrid_rerank",
):
    from src.metrics import StageTimer

    metrics = PipelineMetrics(
        pipeline_variant=pipeline,
        query=question,
        session_id=session_id,
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    docs    = retrieve(question, metrics, session_id, pipeline=pipeline)
    context = format_context(docs)

    if not docs:
        yield {
            "type":    "error",
            "message": "No documents found. Please upload a PDF first."
        }
        return
    
    # ── Sources with scores ───────────────────────────────────────────────────
    sources = [
        {
            "source":         d.metadata.get("source", "unknown"),
            "page":           d.metadata.get("page", "?"),
            "chunk":          d.page_content[:100] + "...",
            "vector_score":   d.metadata.get("vector_score", 0.0),
            "bm25_score":     d.metadata.get("bm25_score", 0.0),
            "reranker_score": d.metadata.get("reranker_score", 0.0),
        }
        for d in docs
    ]

    # ── Generation ────────────────────────────────────────────────────────────
    prompt        = prompt_template.format_messages(
        context=context,
        question=question
    )
    input_tokens  = count_tokens(context + question)
    output_tokens = 0

    with StageTimer() as t:
        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                output_tokens += count_tokens(token)
                yield {
                    "type":  "token",
                    "token": token
                }

    metrics.generation_ms            = t.elapsed_ms
    metrics.tokens.llm_input_tokens  = input_tokens
    metrics.tokens.llm_output_tokens = output_tokens

    cost_tracker.record(metrics)

    # ── Done event ────────────────────────────────────────────────────────────
    yield {
        "type":    "done",
        "pipeline": pipeline,
        "sources": sources,
        "metrics": metrics.summary()
    }

def compare_pipelines(question: str) -> dict:
    """
    Run the same query through all three pipeline variants.
    Useful for building the comparison table for your resume/README.
    """
    results = {}
    for pipeline in ["dense_only", "hybrid", "hybrid_rerank"]:
        print(f"\n{'='*50}")
        print(f"Running pipeline: {pipeline}")
        print(f"{'='*50}")
        results[pipeline] = generate(question, pipeline=pipeline)

    print("\n── Pipeline Comparison ──────────────────────────────")
    print(f"{'Pipeline':<20} {'Latency':>10} {'Tokens':>8} {'Cost':>10}")
    print(f"{'-'*50}")
    for name, result in results.items():
        m = result["metrics"]
        print(
            f"{name:<20}"
            f"{m['latency']['total_ms']:>9.0f}ms"
            f"{m['tokens']['total']:>8}"
            f"  ${m['cost']['total_usd']:>8.6f}"
        )

    return results



if __name__ == "__main__":
    question = "what are the technical skills mentioned?"

    print("\n── Single pipeline test ─────────────────────────────")
    result = generate(question, pipeline="hybrid_rerank")
    
    # latency breakdown
    m = result["metrics"]
    print("\n── Latency breakdown ────────────────────────────────")
    print(f"  Dense retrieval : {m['latency']['dense_retrieval_ms']:.0f}ms")
    print(f"  Sparse retrieval: {m['latency']['sparse_retrieval_ms']:.0f}ms")
    print(f"  RRF fusion      : {m['latency']['rrf_fusion_ms']:.0f}ms")
    print(f"  Rerank          : {m['latency']['rerank_ms']:.0f}ms")
    print(f"  Generation      : {m['latency']['generation_ms']:.0f}ms")
    print(f"  Total           : {m['latency']['total_ms']:.0f}ms")

    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} page {s['page']}: {s['chunk']}")

    print("\n\n── A/B comparison ───────────────────────────────────")
    cost_tracker.report()
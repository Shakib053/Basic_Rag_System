"""
RAGAS Evaluation Module for Personal RAG Assistant

This module provides comprehensive evaluation capabilities using the RAGAS framework
to measure the quality of the RAG pipeline's retrieval and generation components.

Metrics evaluated:
- Faithfulness: How factually consistent the answer is with the retrieved context
- Context Precision: Signal-to-noise ratio of retrieved contexts
- Context Recall: Whether all relevant information was retrieved
- Answer Relevancy: How relevant the answer is to the question
- Answer Correctness: Factual correctness against ground truth (requires reference answers)
"""

from __future__ import annotations

import os
import json
import signal
import sys
import types
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# --- Compatibility shim ---------------------------------------------------
# RAGAS 0.4.3 imports ChatVertexAI from langchain_community.chat_models.vertexai
# at module load time, but that path was removed in langchain-community 0.4.x.
# Redirect to the standalone langchain-google-vertexai package.
import langchain_community  # noqa: E402

_vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
from langchain_google_vertexai import ChatVertexAI  # noqa: E402
_vertexai_mod.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_mod

_llms_mod = sys.modules.get("langchain_community.llms")
if _llms_mod and not hasattr(_llms_mod, "VertexAI"):
    from langchain_google_vertexai import VertexAI  # noqa: E402
    _llms_mod.VertexAI = VertexAI
# --------------------------------------------------------------------------

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
    answer_correctness,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from chat import get_rag_response
from retrieval.hybrid_retrieval import rerank_documents
from retrieval.query_enhancement import plan_queries, unique_documents
from embeddings.text_embeddings import get_text_embedding_model

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EVALUATION_MODEL = "google/gemma-4-31b-it:free"  # Use heavy model for evaluation
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAGAS_TIMEOUT_SECONDS = int(os.getenv("RAGAS_TIMEOUT_SECONDS", "60"))
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "1"))
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
RAG_PIPELINE_TIMEOUT_SECONDS = int(os.getenv("RAG_PIPELINE_TIMEOUT_SECONDS", "120"))


def log_step(message: str) -> None:
    """Print timestamped progress so a slow provider call is easy to locate."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_with_timeout(label: str, timeout_seconds: int, func):
    """Run a blocking stage with a hard timeout."""
    if timeout_seconds <= 0:
        return func()

    def handle_timeout(signum, frame):
        raise TimeoutError(f"{label} timed out after {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass
class EvaluationSample:
    """A single evaluation sample with question, ground truth, and expected contexts."""
    question: str
    ground_truth: str
    expected_context_keywords: Optional[List[str]] = None  # Keywords that should appear in retrieved context


# Ground truth QA pairs derived from the indexed documents
EVALUATION_SAMPLES: List[EvaluationSample] = [
    # Personal Info
    EvaluationSample(
        question="What is Kazi Tanjim Shakib's current job title and employer?",
        ground_truth="Kazi Tanjim Shakib is a Senior Software Engineer at Samsung R&D Institute Bangladesh Ltd.",
        expected_context_keywords=["Senior Software Engineer", "Samsung R&D Institute Bangladesh"],
    ),
    # Travel History
    EvaluationSample(
        question="Which travel destination did Kazi Tanjim Shakib rate 10/10?",
        ground_truth="Bandarban and Saint Martin Island both received a 10/10 rating.",
        expected_context_keywords=["Bandarban", "Saint Martin", "10/10"],
    ),
    # Projects / GitHub
    EvaluationSample(
        question="What is the Salah app?",
        ground_truth="Salah is an open-source, non-profit iOS application for Islamic prayer timings and tracking, built with SwiftUI and Combine.",
        expected_context_keywords=["Salah", "prayer", "Islamic", "SwiftUI", "Combine", "open-source"],
    ),
]


def load_evaluation_samples(json_path: Optional[str] = None) -> List[EvaluationSample]:
    """Load evaluation samples from JSON file or use built-in defaults."""
    if json_path and Path(json_path).exists():
        with open(json_path) as f:
            data = json.load(f)
        return [
            EvaluationSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                expected_context_keywords=item.get("expected_context_keywords"),
            )
            for item in data.get("samples", [])
        ]
    return EVALUATION_SAMPLES


def build_evaluation_dataset(
    json_path: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> EvaluationDataset:
    """
    Run the RAG pipeline on each evaluation question and build a RAGAS EvaluationDataset.

    Args:
        json_path: Optional path to JSON file with evaluation samples
        sample_limit: Optional number of samples to run before building the dataset

    Returns:
        EvaluationDataset: Dataset with SingleTurnSample objects containing
        user_input, response, retrieved_contexts, and reference (ground_truth)
    """
    eval_samples = load_evaluation_samples(json_path)
    if sample_limit:
        eval_samples = eval_samples[:sample_limit]
        log_step(f"Limited to first {sample_limit} samples for testing")

    log_step(f"Building evaluation dataset by running RAG pipeline on {len(eval_samples)} questions...")

    samples: List[SingleTurnSample] = []
    chat_history = []  # Fresh history for each evaluation

    for idx, eval_sample in enumerate(eval_samples):
        log_step(f"\n[{idx + 1}/{len(eval_samples)}] Evaluating: {eval_sample.question[:60]}...")

        # Get RAG response (this runs the full pipeline: rewrite -> retrieve -> rerank -> generate)
        try:
            started = time.perf_counter()
            log_step("  Generating RAG response...")
            response = run_with_timeout(
                "RAG response generation",
                RAG_PIPELINE_TIMEOUT_SECONDS,
                lambda: get_rag_response(eval_sample.question, chat_history),
            )
            log_step(f"  RAG response generated in {time.perf_counter() - started:.1f}s")
        except Exception as e:
            log_step(f"  ERROR generating response: {e}")
            response = "ERROR: Failed to generate response"

        # Get retrieved contexts for context-based metrics
        # We need to manually run retrieval to capture the contexts
        from chat import retrieval_llm, FINAL_CONTEXT_DOCS, get_hybrid_docs

        try:
            started = time.perf_counter()
            log_step("  Rewriting query for context capture...")
            query_plan = run_with_timeout(
                "query planning",
                RAG_PIPELINE_TIMEOUT_SECONDS,
                lambda: plan_queries(eval_sample.question, chat_history, retrieval_llm),
            )
            log_step("  Retrieving text contexts...")
            candidate_groups = run_with_timeout(
                "text context retrieval",
                RAG_PIPELINE_TIMEOUT_SECONDS,
                lambda: [get_hybrid_docs(query) for query in query_plan.queries],
            )
            candidate_docs = unique_documents(candidate_groups)
            log_step("  Reranking text contexts...")
            docs = run_with_timeout(
                "text context reranking",
                RAG_PIPELINE_TIMEOUT_SECONDS,
                lambda: rerank_documents(query_plan.queries[-1], candidate_docs, top_k=FINAL_CONTEXT_DOCS),
            )
            log_step(f"  Context capture completed in {time.perf_counter() - started:.1f}s")

            # Extract context texts for RAGAS
            retrieved_contexts = [doc.page_content for doc in docs]
        except Exception as e:
            log_step(f"  ERROR retrieving contexts: {e}")
            retrieved_contexts = []

        # Create SingleTurnSample for RAGAS
        sample = SingleTurnSample(
            user_input=eval_sample.question,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=eval_sample.ground_truth,
        )
        samples.append(sample)

        log_step(f"  Response: {response[:100]}...")
        log_step(f"  Contexts retrieved: {len(retrieved_contexts)}")

    dataset = EvaluationDataset(samples=samples)
    log_step(f"\n✓ Built evaluation dataset with {len(samples)} samples")
    return dataset


def configure_ragas_llm_and_embeddings() -> Dict[str, Any]:
    """
    Configure RAGAS to use custom LLM (Ollama or OpenRouter via ChatOpenAI) and embeddings.

    Returns:
        Dict with 'llm' and 'embeddings' keys for passing to evaluate()
    """
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if llm_provider == "ollama":
        eval_llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.0,
            client_kwargs={"timeout": RAGAS_TIMEOUT_SECONDS},
        )
    else:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for RAGAS evaluation with OpenRouter")

        eval_llm = ChatOpenAI(
            model=EVALUATION_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=1024,
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,
        )

    # Embeddings for RAGAS (used by context_precision/recall for semantic matching)
    eval_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return {"llm": eval_llm, "embeddings": eval_embeddings}


def run_ragas_evaluation(
    dataset: EvaluationDataset,
    metrics: Optional[List] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on the dataset with specified metrics.

    Args:
        dataset: EvaluationDataset with SingleTurnSample objects
        metrics: List of RAGAS metrics to evaluate (defaults to all core metrics)
        output_file: Optional path to save results JSON

    Returns:
        Dict with metric scores and detailed results
    """
    if metrics is None:
        metrics = [
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
            answer_correctness,
        ]

    # Configure custom LLM and embeddings
    ragas_config = configure_ragas_llm_and_embeddings()

    print("\n" + "=" * 60)
    print("Running RAGAS Evaluation")
    print("=" * 60)
    print(f"Model: {EVALUATION_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL_NAME}")
    print(f"Metrics: {[m.name for m in metrics]}")
    print(f"Samples: {len(dataset.samples)}")
    print("=" * 60 + "\n")

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_config["llm"],
        embeddings=ragas_config["embeddings"],
        run_config=RunConfig(
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,
            max_workers=RAGAS_MAX_WORKERS,
        ),
        batch_size=1,
    )

    # Convert to dictionary for easier handling
    result_dict = result.to_pandas().to_dict(orient="records")

    # Calculate aggregate scores
    aggregate_scores = {}
    for metric in metrics:
        metric_name = metric.name
        scores = [r.get(metric_name, 0) for r in result_dict if metric_name in r]
        if scores:
            aggregate_scores[metric_name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for metric_name, stats in aggregate_scores.items():
        print(f"{metric_name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
    print("=" * 60)

    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        full_results = {
            "aggregate_scores": aggregate_scores,
            "per_sample_results": result_dict,
            "metadata": {
                "model": EVALUATION_MODEL,
                "embeddings": EMBEDDING_MODEL_NAME,
                "num_samples": len(dataset.samples),
                "metrics": [m.name for m in metrics],
            }
        }

        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    return {
        "aggregate_scores": aggregate_scores,
        "per_sample_results": result_dict,
    }


def run_faithfulness_only_evaluation(
    dataset: EvaluationDataset,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run only faithfulness evaluation (reference-free, faster)."""
    return run_ragas_evaluation(dataset, metrics=[faithfulness], output_file=output_file)


def run_retrieval_only_evaluation(
    dataset: EvaluationDataset,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run only retrieval metrics (context_precision, context_recall - need ground truth)."""
    return run_ragas_evaluation(
        dataset,
        metrics=[context_precision, context_recall],
        output_file=output_file
    )


def run_generation_only_evaluation(
    dataset: EvaluationDataset,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run only generation metrics (faithfulness, answer_relevancy - reference-free for faithfulness)."""
    return run_ragas_evaluation(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        output_file=output_file
    )


def run_full_evaluation(output_file: str = "evaluation_results.json") -> Dict[str, Any]:
    """Run complete RAGAS evaluation pipeline."""
    dataset = build_evaluation_dataset()
    return run_ragas_evaluation(dataset, output_file=output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "faithfulness", "retrieval", "generation"],
        default="full",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--dataset",
        default="eval_dataset.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Run evaluation on only the first N samples (for quick testing)",
    )

    args = parser.parse_args()

    # Build dataset from JSON file
    dataset = build_evaluation_dataset(args.dataset, sample_limit=args.sample)

    # Run selected evaluation mode
    if args.mode == "full":
        run_ragas_evaluation(dataset, output_file=args.output)
    elif args.mode == "faithfulness":
        run_faithfulness_only_evaluation(dataset, output_file=args.output)
    elif args.mode == "retrieval":
        run_retrieval_only_evaluation(dataset, output_file=args.output)
    elif args.mode == "generation":
        run_generation_only_evaluation(dataset, output_file=args.output)

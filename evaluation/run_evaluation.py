"""Run the agent over the evaluation dataset and score it with RAGAS.

Usage:  python evaluation/run_evaluation.py

Produces evaluation/results.csv (per-question scores) and prints a summary.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from agentic_rag.config import settings
from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index, get_embeddings

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.csv"


def collect_samples(graph, cases: list[dict]) -> list[SingleTurnSample]:
    """Run every eval question through the agent and record its traces."""
    samples = []

    for i, case in enumerate(cases, start=1):
        start = time.perf_counter()

        result = graph.invoke({
            "question": case["question"]
        })

        elapsed = time.perf_counter() - start

        print(
            f"[{i}/{len(cases)}] "
            f"({elapsed:.1f}s) "
            f"{case['question']}"
        )

        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=result["generation"],
                retrieved_contexts=[
                    doc.page_content
                    for doc in result.get("documents", [])
                ],
                reference=case["ground_truth"],
            )
        )

    return samples


def main() -> None:
    cases = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    print("== Running the agent over the evaluation dataset ==")

    ensure_index()

    samples = collect_samples(
        build_graph(),
        cases,
    )

    print("\n== Scoring with RAGAS ==")

    # Groq is used as the LLM judge for RAGAS evaluation.
    judge = LangchainLLMWrapper(
    ChatGroq(
        model=settings.eval_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=2048,
        n=1,
    )
)
    # Embeddings remain local SentenceTransformer embeddings.
    embeddings = LangchainEmbeddingsWrapper(
        get_embeddings()
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embeddings,

        # Keep evaluation requests serial to avoid unnecessary
        # concurrent API calls and rate-limit issues.
        run_config=RunConfig(
            timeout=600,
            max_workers=1,
        ),
    )

    df = result.to_pandas()

    df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n== Average scores ==")

    metric_columns = df.select_dtypes("number").columns

    for metric in metric_columns:
        print(
            f"  {metric:35s} "
            f"{df[metric].mean():.3f}"
        )

    print(
        f"\nPer-question results saved to "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
"""
Offline evaluation pipeline — Phase 3.

For each question in the golden set:
  1. Run the full RAG pipeline
  2. Ask Groq to judge whether the answer is grounded in the retrieved chunks
  3. Compute a faithfulness score
  4. Write a report
  5. Exit with code 1 if below threshold (CI gate)
"""

import json
import sys
from pathlib import Path
from loguru import logger
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

FAITHFULNESS_THRESHOLD = 0.70  # CI fails if mean score drops below this


JUDGE_PROMPT = """You are an evaluation judge for a RAG system.

Your job: decide whether the generated answer is faithful to the source chunks.
Faithful means every claim in the answer is directly supported by the sources.
The answer does NOT need to be complete — it just must not contradict or go beyond the sources.

Respond with JSON only, no markdown:
{
  "score": 0.0 to 1.0,
  "reasoning": "one sentence explanation"
}

Score guide:
  1.0 — every claim is clearly supported by the sources
  0.75 — mostly supported, one minor unsupported claim
  0.5  — some claims supported, some not
  0.25 — mostly unsupported or contradicts sources
  0.0  — completely unsupported or hallucinated
"""


def judge_faithfulness(
    question: str,
    answer: str,
    chunks: list[dict],
    client: Groq,
) -> tuple[float, str]:
    """
    Use Groq as an LLM judge to score faithfulness.
    Returns (score, reasoning).
    """
    # If the pipeline said insufficient evidence, that's always faithful
    if answer.startswith("INSUFFICIENT_EVIDENCE"):
        return 1.0, "Correctly declined to answer with insufficient evidence"

    sources_text = "\n\n---\n\n".join(
    f"[Source {i+1}]: {c.get('text', c.get('excerpt', ''))[:600]}"
    for i, c in enumerate(chunks)
    )

    user_message = f"""Question: {question}

Generated answer:
{answer}

Source chunks the answer was based on:
{sources_text}

Score the faithfulness of the answer to the sources."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,      # zero temperature for consistent scoring
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()

    try:
        # strip markdown fences if model added them anyway
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return float(data["score"]), data["reasoning"]
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Could not parse judge response: {raw}")
        return 0.0, f"Parse error: {raw}"


def run_evaluation(pipeline) -> dict:
    """
    Run the full evaluation against the golden set.
    Returns a report dict.
    """
    golden_path = Path("data/golden_set.json")
    if not golden_path.exists():
        raise FileNotFoundError(
            "data/golden_set.json not found. Create it before running eval."
        )

    with open(golden_path) as f:
        golden_set = json.load(f)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    results = []
    logger.info(f"Running evaluation on {len(golden_set)} questions...")
    logger.info("=" * 60)

    for i, example in enumerate(golden_set, 1):
        question = example["question"]
        logger.info(f"[{i}/{len(golden_set)}] {question}")

        try:
            # Run the full pipeline
            result = pipeline.query(question)
            answer = result["answer"]
            chunks = result["sources"]

            # Judge faithfulness
            score, reasoning = judge_faithfulness(question, answer, chunks, client)

        except Exception as e:
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                logger.warning(f"Rate limit hit on question {i}, skipping...")
                continue   # skip this question entirely, don't score it zero
            logger.error(f"Pipeline error: {e}")
            answer = f"ERROR: {e}"
            score = 0.0
            reasoning = error_str

        passed = score >= FAITHFULNESS_THRESHOLD
        status = "✅ PASS" if passed else "❌ FAIL"

        logger.info(f"  Score: {score:.2f} | {status} | {reasoning}")

        results.append({
            "question": question,
            "expected_answer": example["expected_answer"],
            "generated_answer": answer,
            "faithfulness_score": score,
            "reasoning": reasoning,
            "passed": passed,
            "source_paper": example.get("source_paper", ""),
        })

    # Compute summary
    scores = [r["faithfulness_score"] for r in results]

    if not scores:
        logger.error("No questions completed — all hit rate limits. Wait and retry.")
        sys.exit(2)

    mean_score = sum(scores) / len(scores)
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    passed_ci = mean_score >= FAITHFULNESS_THRESHOLD

    report = {
        "summary": {
            "total_questions": len(results),
            "mean_faithfulness": round(mean_score, 4),
            "pass_rate": round(pass_rate, 4),
            "threshold": FAITHFULNESS_THRESHOLD,
            "passed_ci": passed_ci,
        },
        "results": results,
    }

    # Save report
    report_path = Path("data/eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Mean faithfulness: {mean_score:.4f}")
    logger.info(f"Pass rate:         {pass_rate:.1%}")
    logger.info(f"CI result:         {'✅ PASSED' if passed_ci else '❌ FAILED'}")
    logger.info(f"Report saved to:   {report_path}")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    papers = [
        ("data/raw/attention.pdf",  "Attention Is All You Need"),
        ("data/raw/bert.pdf",       "BERT"),
        ("data/raw/rag_paper.pdf",  "Retrieval-Augmented Generation"),
    ]
    for path, title in papers:
        pipeline.ingest(path, title)

    report = run_evaluation(pipeline)

    # Exit code 1 fails the CI build
    if not report["summary"]["passed_ci"]:
        logger.error("CI FAILED — mean faithfulness below threshold")
        sys.exit(1)

    logger.success("CI PASSED")
    sys.exit(0)
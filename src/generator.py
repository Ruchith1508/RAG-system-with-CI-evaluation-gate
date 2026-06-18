import os
from groq import Groq
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


SYSTEM_PROMPT = """You are a precise research assistant. You answer questions about
scientific papers using ONLY the provided sources below.

Rules you must follow:
1. Every factual claim must have an inline citation like [Source 1] or [Source 3].
2. Only use information from the provided sources — never your own knowledge.
3. If the sources don't contain enough information to answer, respond with exactly:
   INSUFFICIENT_EVIDENCE: <brief reason why>
4. Never guess, speculate, or fill gaps with plausible-sounding content.
5. Be concise and precise.
6. Do NOT add any details, numbers, or specifics that are not explicitly stated
   in the sources — even if you know them to be true."""


def format_sources(results: list[dict]) -> str:
    """
    Turn retrieved chunks into a numbered source block for the prompt.
    This is what the LLM will read and cite from.
    """
    blocks = []
    for i, result in enumerate(results, start=1):
        block = (
            f"[Source {i}]\n"
            f"Paper: {result['metadata']['doc_title']}\n"
            f"Relevance score: {result['score']}\n\n"
            f"{result['text']}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Send the question + retrieved chunks to Groq and get a cited answer back.
    Returns a dict with the answer text and the sources used.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    sources_block = format_sources(retrieved_chunks)

    user_message = f"""Question: {question}

<sources>
{sources_block}
</sources>

Answer using only the sources above. Cite every claim with [Source N]."""

    logger.info(f"Sending question to Groq: '{question}'")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,   # low temperature = more faithful, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # Check if the model signaled insufficient evidence
    sufficient = not answer.startswith("INSUFFICIENT_EVIDENCE")

    return {
        "question": question,
        "answer": answer,
        "sufficient_evidence": sufficient,
        "sources": [
            {
                "index": i + 1,
                "doc_title": r["metadata"]["doc_title"],
                "chunk_index": r["metadata"]["chunk_index"],
                "score": r["score"],
                "excerpt": r["text"][:200],
            }
            for i, r in enumerate(retrieved_chunks)
        ],
    }


def pretty_print(result: dict) -> None:
    """Print a result in a readable format."""
    print("\n" + "="*60)
    print(f"Q: {result['question']}")
    print("="*60)
    print(f"\n{result['answer']}\n")

    if result["sufficient_evidence"]:
        print("--- Sources used ---")
        for s in result["sources"]:
            print(f"  [Source {s['index']}] {s['doc_title']}, chunk {s['chunk_index']} (score: {s['score']})")
            print(f"             \"{s['excerpt'][:100]}...\"")
    print("="*60)


if __name__ == "__main__":
    from src.vector_store import VectorStore

    store = VectorStore()

    # Make sure we have chunks indexed
    if store.count() == 0:
        from src.ingestion import load_pdf, chunk_text
        text = load_pdf("data/raw/attention.pdf")
        chunks = chunk_text(text, "Attention Is All You Need", "data/raw/attention.pdf")
        store.add_chunks(chunks)

    # Test 3 different questions
    questions = [
        "How does multi-head attention work?",
        "What BLEU score did the Transformer achieve on English to German translation?",
        "Who invented the iPhone?",  # this should trigger INSUFFICIENT_EVIDENCE
    ]

    for question in questions:
        chunks = store.search(question, top_k=5)
        result = generate_answer(question, chunks)
        pretty_print(result)
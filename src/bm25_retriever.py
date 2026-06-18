import re
from rank_bm25 import BM25Okapi
from loguru import logger
from src.ingestion import Chunk


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Lowercase, remove punctuation, split on whitespace.
    BM25 works on token lists — the cleaner the better.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    Keyword-based retriever using Okapi BM25.
    Built entirely in memory from your chunk list.
    """

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

        # BM25 needs a list of token lists — one per document
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(f"BM25 index built over {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top_k chunks most relevant to the query by keyword match.
        Returns the same format as VectorStore.search() so they're interchangeable.
        """
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # pair each chunk with its score, sort descending
        scored = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in scored:
            if score > 0:  # skip chunks with zero keyword overlap
                results.append({
                    "text": self._chunks[idx].text,
                    "metadata": {
                        "doc_title": self._chunks[idx].doc_title,
                        "source_file": self._chunks[idx].source_file,
                        "chunk_index": self._chunks[idx].chunk_index,
                        "token_count": self._chunks[idx].token_count,
                    },
                    "score": round(float(score), 4),
                })

        return results


if __name__ == "__main__":
    from src.ingestion import load_pdf, chunk_text

    # Load and chunk the paper
    text = load_pdf("data/raw/attention.pdf")
    chunks = chunk_text(text, "Attention Is All You Need", "data/raw/attention.pdf")

    retriever = BM25Retriever(chunks)

    # Test 1: exact term search — BM25 should shine here
    q1 = "BLEU score 28.4 English German translation"
    print(f"\nQuery: '{q1}'")
    print("-" * 50)
    for r in retriever.search(q1, top_k=3):
        print(f"  Score: {r['score']} | Chunk {r['metadata']['chunk_index']}")
        print(f"  {r['text'][:200]}\n")

    # Test 2: conceptual search — vector search would do better here
    q2 = "how does the model understand word positions"
    print(f"\nQuery: '{q2}'")
    print("-" * 50)
    for r in retriever.search(q2, top_k=3):
        print(f"  Score: {r['score']} | Chunk {r['metadata']['chunk_index']}")
        print(f"  {r['text'][:200]}\n")
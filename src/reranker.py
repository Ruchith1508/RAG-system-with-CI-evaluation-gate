from sentence_transformers import CrossEncoder
from loguru import logger


class Reranker:
    """
    Cross-encoder reranker.
    Takes (query, chunk) pairs and scores them together —
    much more accurate than embedding-based similarity.
    Model: ms-marco-MiniLM-L-6-v2
      - Trained specifically on passage relevance ranking
      - Small and fast enough to run locally
      - Outputs a raw relevance score (higher = more relevant)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name)
        logger.success("Reranker ready")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Rescore chunks by reading (query, chunk) together.
        Returns top_k chunks sorted by rerank score.
        """
        if not chunks:
            return []

        # Build pairs: [[query, chunk1_text], [query, chunk2_text], ...]
        pairs = [[query, chunk["text"]] for chunk in chunks]

        logger.info(f"Reranking {len(pairs)} candidates...")
        scores = self._model.predict(pairs)

        # attach rerank score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        # sort by rerank score descending
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Top rerank scores: "
            f"{[r['rerank_score'] for r in reranked[:top_k]]}"
        )

        return reranked[:top_k]


if __name__ == "__main__":
    from src.ingestion import load_pdf, chunk_text
    from src.vector_store import VectorStore
    from src.hybrid_retriever import HybridRetriever

    # Load and chunk
    text = load_pdf("data/raw/attention.pdf")
    chunks = chunk_text(text, "Attention Is All You Need", "data/raw/attention.pdf")

    # Retrieve with hybrid search
    store = VectorStore()
    hybrid = HybridRetriever(chunks, store)
    reranker = Reranker()

    query = "multi-head attention mechanism"

    print(f"\nQuery: '{query}'")

    # Get 10 candidates from hybrid search
    candidates = hybrid.search(query, top_k=10)

    print(f"\n--- BEFORE reranking (hybrid RRF order) ---")
    for i, c in enumerate(candidates, 1):
        print(f"  #{i} | RRF: {c['rrf_score']} | Chunk {c['metadata']['chunk_index']} | {c['text'][:80].strip()}")

    # Rerank and take top 5
    reranked = reranker.rerank(query, candidates, top_k=5)

    print(f"\n--- AFTER reranking (cross-encoder order) ---")
    for i, c in enumerate(reranked, 1):
        print(f"  #{i} | Rerank: {c['rerank_score']} | Chunk {c['metadata']['chunk_index']} | {c['text'][:80].strip()}")

    print("\n--- Key difference ---")
    before_ids = [c['metadata']['chunk_index'] for c in candidates[:5]]
    after_ids  = [c['metadata']['chunk_index'] for c in reranked]
    print(f"  Before top-5 chunks: {before_ids}")
    print(f"  After  top-5 chunks: {after_ids}")
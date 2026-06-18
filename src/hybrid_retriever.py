from loguru import logger
from src.ingestion import Chunk
from src.bm25_retriever import BM25Retriever
from src.vector_store import VectorStore


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[dict]:
    """
    Combine two ranked lists into one using Reciprocal Rank Fusion.

    k=60         — dampening constant (standard default, rarely needs tuning)
    bm25_weight  — how much to trust keyword results (0.4)
    vector_weight — how much to trust semantic results (0.6)

    We weight vector slightly higher because semantic understanding
    is generally more reliable than keyword overlap for research papers.
    """
    # We need a stable key to identify each chunk across both lists.
    # We use the chunk_index from metadata since it's unique per document.
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    def _add_list(results: list[dict], weight: float):
        for rank, result in enumerate(results, start=1):
            # unique key for this chunk
            key = f"{result['metadata']['source_file']}::chunk_{result['metadata']['chunk_index']}"

            rrf_score = weight * (1.0 / (k + rank))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score

            # store the chunk data the first time we see it
            if key not in chunk_map:
                chunk_map[key] = result

    _add_list(bm25_results, bm25_weight)
    _add_list(vector_results, vector_weight)

    # sort all chunks by their combined RRF score
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    fused = []
    for key in sorted_keys:
        result = chunk_map[key].copy()
        result["rrf_score"] = round(rrf_scores[key], 6)
        # keep original scores for debugging
        fused.append(result)

    return fused


class HybridRetriever:
    """
    Combines BM25 + vector search via RRF.
    This is the retrieval layer your pipeline will actually use going forward.
    """

    def __init__(self, chunks: list[Chunk], vector_store: VectorStore):
        self._bm25 = BM25Retriever(chunks)
        self._vector_store = vector_store
        self._chunks = chunks

    def search(self, query: str, top_k: int = 5, initial_k: int = 20) -> list[dict]:
        """
        1. Get top initial_k results from both BM25 and vector search
        2. Fuse them with RRF
        3. Return the top_k fused results

        We cast a wide net (20) before fusion, then narrow to top_k.
        This gives RRF enough candidates to work with.
        """
        logger.info(f"Hybrid search for: '{query}'")

        bm25_results = self._bm25.search(query, top_k=initial_k)
        vector_results = self._vector_store.search(query, top_k=initial_k)

        logger.debug(f"  BM25 returned {len(bm25_results)} results")
        logger.debug(f"  Vector returned {len(vector_results)} results")

        fused = reciprocal_rank_fusion(bm25_results, vector_results)

        logger.info(f"  Fused into {len(fused)} unique chunks, returning top {top_k}")
        return fused[:top_k]


if __name__ == "__main__":
    from src.ingestion import load_pdf, chunk_text

    text = load_pdf("data/raw/attention.pdf")
    chunks = chunk_text(text, "Attention Is All You Need", "data/raw/attention.pdf")

    store = VectorStore()
    retriever = HybridRetriever(chunks, store)

    queries = [
        "BLEU score 28.4 English German translation",
        "how does the model understand word positions",
        "multi-head attention mechanism",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        results = retriever.search(query, top_k=5)
        for i, r in enumerate(results, 1):
            print(
                f"  #{i} | RRF: {r['rrf_score']} | "
                f"Chunk {r['metadata']['chunk_index']} | "
                f"{r['text'][:120].strip()}"
            )
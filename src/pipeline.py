"""
The complete RAG pipeline in one place:
  load → chunk → embed → hybrid retrieve → rerank → generate

This is what you call when a user asks a question.
"""

from loguru import logger
from src.ingestion import load_pdf, chunk_text, Chunk
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import generate_answer, pretty_print


class RAGPipeline:
    def __init__(self):
        logger.info("Initialising RAG pipeline...")
        self._store = VectorStore()
        self._chunks: list[Chunk] = []
        self._retriever: HybridRetriever | None = None
        self._reranker = Reranker()
        logger.success("Pipeline ready")

    def ingest(self, file_path: str, title: str) -> int:
        """
        Load a PDF, chunk it, embed and store it.
        Returns number of chunks created.
        Call this once per document before querying.
        """
        text = load_pdf(file_path)
        new_chunks = chunk_text(text, title, file_path)

        # Only add chunks not already in the store
        existing_ids = {
            f"{c.source_file}::chunk_{c.chunk_index}"
            for c in self._chunks
        }
        fresh = [
            c for c in new_chunks
            if f"{c.source_file}::chunk_{c.chunk_index}" not in existing_ids
        ]

        if fresh:
            self._store.add_chunks(fresh)
            self._chunks.extend(fresh)
            # Rebuild the retriever with updated chunk list
            self._retriever = HybridRetriever(self._chunks, self._store)
            logger.success(f"Ingested '{title}': {len(fresh)} new chunks")
        else:
            logger.info(f"'{title}' already indexed, skipping")
            if self._retriever is None:
                self._retriever = HybridRetriever(self._chunks, self._store)

        return len(fresh)

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a question using the full pipeline:
        hybrid retrieve → rerank → generate with citations
        """
        if not self._retriever:
            raise RuntimeError("No documents ingested yet. Call pipeline.ingest() first.")

        # 1. Hybrid retrieval: get top 20 candidates
        candidates = self._retriever.search(question, top_k=20)

        # 2. Rerank: cross-encoder picks best 5
        best_chunks = self._reranker.rerank(question, candidates, top_k=top_k)

        # 3. Generate: send to Groq with citation instructions
        result = generate_answer(question, best_chunks)
        return result


if __name__ == "__main__":
    pipeline = RAGPipeline()

    # Ingest all three papers
    papers = [
        ("data/raw/attention.pdf",  "Attention Is All You Need"),
        ("data/raw/bert.pdf",       "BERT"),
        ("data/raw/rag_paper.pdf",  "Retrieval-Augmented Generation"),
    ]

    for path, title in papers:
        pipeline.ingest(path, title)

    print(f"\nTotal chunks indexed: {pipeline._store.count()}")

    # Ask questions that span multiple papers
    questions = [
        "What is the Transformer architecture and how does attention work?",
        "How does BERT use bidirectional training?",
        "What are the key differences between RAG and standard language models?",
        "Who invented the telephone?",  # should be INSUFFICIENT_EVIDENCE
    ]

    for question in questions:
        result = pipeline.query(question)
        pretty_print(result)
        input("\nPress Enter for next question...")
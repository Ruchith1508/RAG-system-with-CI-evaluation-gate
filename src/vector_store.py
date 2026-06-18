import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from src.ingestion import Chunk


class VectorStore:
    """
    Wraps ChromaDB. Responsible for:
    - storing chunks as embeddings
    - searching for the most relevant chunks given a query
    """

    def __init__(self, persist_directory: str = "data/chroma_db"):
        # PersistentClient saves to disk so your embeddings
        # survive between runs — you won't re-embed every time
        self._client = chromadb.PersistentClient(path=persist_directory)

        # This is the embedding model — it runs locally on your machine
        # all-MiniLM-L6-v2 is small, fast, and good enough for Phase 1
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # A "collection" is like a table in a database
        self._collection = self._client.get_or_create_collection(
            name="research_papers",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

        logger.info(
            f"VectorStore ready — {self._collection.count()} chunks already indexed"
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store a list of chunks."""
        if not chunks:
            return

        # ChromaDB needs three parallel lists: ids, texts, and metadata
        ids = [f"{c.source_file}::chunk_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_title": c.doc_title,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        # This is where the embedding model runs — may take a few seconds
        logger.info(f"Embedding {len(chunks)} chunks (this may take a moment)...")
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.success(f"Stored {len(chunks)} chunks in ChromaDB")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top_k most relevant chunks for a given query.
        Returns a list of dicts with 'text', 'metadata', and 'score'.
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's response format
        chunks_out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks_out.append({
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # convert distance → similarity
            })

        return chunks_out

    def count(self) -> int:
        return self._collection.count()


if __name__ == "__main__":
    from src.ingestion import load_pdf, chunk_text

    # 1. Load and chunk the paper
    text = load_pdf("data/raw/attention.pdf")
    chunks = chunk_text(text, "Attention Is All You Need", "data/raw/attention.pdf")

    # 2. Store in ChromaDB
    store = VectorStore()

    if store.count() == 0:
        store.add_chunks(chunks)
    else:
        logger.info("Chunks already indexed, skipping embedding step")

    # 3. Test a search
    query = "How does the multi-head attention mechanism work?"
    logger.info(f"\nSearching for: '{query}'")
    results = store.search(query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']}) ---")
        print(f"From: {result['metadata']['doc_title']}, chunk {result['metadata']['chunk_index']}")
        print(result["text"][:400])
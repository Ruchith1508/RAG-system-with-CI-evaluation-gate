from pathlib import Path
from pypdf import PdfReader
from loguru import logger
import re
import tiktoken
from dataclasses import dataclass

@dataclass
class Chunk:
    """
    A single piece of a document, ready to be embedded and stored.
    We keep the metadata alongside the text so we never lose track
    of where a chunk came from.
    """
    text: str
    doc_title: str
    source_file: str
    chunk_index: int
    token_count: int

def load_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF file.
    Returns a single clean string of all the text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")

    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
            logger.debug(f"  Page {i+1}: extracted {len(text)} characters")

    full_text = "\n\n".join(pages)

    # Clean up: collapse 3+ newlines into 2, strip leading/trailing whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    logger.success(f"Done — {len(reader.pages)} pages, {len(full_text)} characters")
    return full_text

def chunk_text(
    text: str,
    doc_title: str,
    source_file: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    Split text into overlapping token-aware chunks.

    chunk_size    — how many tokens per chunk (600 is a good default)
    chunk_overlap — how many tokens to repeat between chunks (100 is a good default)
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Convert the entire text into a list of token IDs
    all_tokens = tokenizer.encode(text)
    total_tokens = len(all_tokens)

    logger.info(f"Chunking '{doc_title}': {total_tokens} total tokens")

    chunks = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        # grab chunk_size tokens starting at `start`
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = all_tokens[start:end]

        # decode token IDs back into readable text
        chunk_text_str = tokenizer.decode(chunk_tokens)

        chunk = Chunk(
            text=chunk_text_str,
            doc_title=doc_title,
            source_file=source_file,
            chunk_index=chunk_index,
            token_count=len(chunk_tokens),
        )
        chunks.append(chunk)

        logger.debug(f"  Chunk {chunk_index}: tokens {start}→{end} ({len(chunk_tokens)} tokens)")

        # move forward by (chunk_size - chunk_overlap) so the next chunk
        # starts 100 tokens BACK from where this one ended
        start += chunk_size - chunk_overlap
        chunk_index += 1

    logger.success(f"Created {len(chunks)} chunks from '{doc_title}'")
    return chunks

if __name__ == "__main__":
    # Step 2 test — load the PDF
    text = load_pdf("data/raw/attention.pdf")

    # Step 3 test — chunk it
    chunks = chunk_text(
        text=text,
        doc_title="Attention Is All You Need",
        source_file="data/raw/attention.pdf",
    )

    # Inspect the results
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\n--- Chunk 0 ---")
    print(chunks[0].text)
    print(f"\n--- Chunk 1 ---")
    print(chunks[1].text[:300])
    print("\n... (notice how chunk 1 starts with the last ~100 tokens of chunk 0)")
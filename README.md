# RAG System — Research Papers

A production-grade Retrieval-Augmented Generation (RAG) system built from scratch, capable of answering questions about research papers with inline citations and automatic quality evaluation.

## What This Does

Upload research papers (PDF), ask questions, get cited answers grounded in the actual documents. The system refuses to answer when the retrieved evidence doesn't support a confident response — no hallucination.

**Live example:**
> Q: What BLEU score did the Transformer achieve on English-German translation?
>
> A: The Transformer big model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task [Source 1], outperforming all previously reported models including ensembles [Source 2].

---

## Architecture

```
PDF → extract text → chunk (600 tokens, 100 overlap) → embed → ChromaDB
                                                                    ↓
Question → BM25 search ──┐
                          ├── RRF Fusion → Cross-Encoder Rerank → Groq LLM → Cited Answer
Question → Vector search ─┘
```

### Retrieval Pipeline
| Stage | What it does |
|---|---|
| **BM25** | Keyword search — finds exact terms and numbers |
| **Vector search** | Semantic search — finds meaning even without matching words |
| **RRF Fusion** | Combines both ranked lists mathematically (no score scaling needed) |
| **Cross-encoder rerank** | Reads (query + chunk) together to eliminate irrelevant results |
| **Top 5 → LLM** | Only the best chunks reach the language model |

---

## Tech Stack

| Component | Library |
|---|---|
| PDF ingestion | pypdf |
| Tokenization | tiktoken |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| Keyword search | rank-bm25 |
| Reranking | sentence-transformers (ms-marco-MiniLM-L-6-v2) |
| LLM | Groq (llama-3.3-70b-versatile) |
| Evaluation | LLM-as-judge (faithfulness scoring) |
| CI | GitHub Actions |

---

## Project Structure

```
rag-research/
├── src/
│   ├── ingestion.py        # PDF loading + token-aware chunking
│   ├── vector_store.py     # ChromaDB wrapper + semantic search
│   ├── bm25_retriever.py   # BM25 keyword search
│   ├── hybrid_retriever.py # RRF fusion of BM25 + vector
│   ├── reranker.py         # Cross-encoder reranking
│   ├── generator.py        # Groq LLM + citation enforcement
│   ├── pipeline.py         # Full pipeline wired together
│   └── evaluator.py        # Faithfulness eval + CI gate
├── data/
│   ├── raw/                # Drop your PDFs here
│   └── golden_set.json     # Verified Q&A pairs for evaluation
├── .github/
│   └── workflows/
│       └── eval.yml        # CI pipeline
├── .env.example
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-research.git
cd rag-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 3. Add documents

Drop PDF files into `data/raw/`. The repo includes three foundational AI papers as examples:
- Attention Is All You Need (Vaswani et al., 2017)
- BERT (Devlin et al., 2018)
- Retrieval-Augmented Generation (Lewis et al., 2020)

### 4. Run the pipeline

```bash
python3 -m src.pipeline
```

### 5. Run evaluation

```bash
python3 -m src.evaluator
```

---

## How Each Component Works

### Step 1 — PDF Ingestion (`src/ingestion.py`)
Loads a PDF page by page using `pypdf`, joins the text, and cleans up excessive whitespace. Supports PDF, Markdown, and plain text.

### Step 2 — Chunking (`src/ingestion.py`)
Splits the full document text into overlapping token windows using `tiktoken`. Each chunk is 600 tokens with 100 tokens of overlap between consecutive chunks. The overlap ensures that sentences split at a boundary aren't lost — the tail of one chunk becomes the head of the next.

### Step 3 — Embeddings + Vector Store (`src/vector_store.py`)
Each chunk is passed through `all-MiniLM-L6-v2`, a local sentence embedding model, which converts text into a 384-dimensional vector. Similar meaning = similar vectors = close together in cosine space. Vectors are stored in ChromaDB with persistence so they survive between runs.

### Step 4 — BM25 Keyword Search (`src/bm25_retriever.py`)
An in-memory Okapi BM25 index built over all chunks. Tokenizes on whitespace after lowercasing and stripping punctuation. Returns chunks ranked by keyword overlap with the query — excellent for exact terms, numbers, and acronyms that vector search can miss.

### Step 5 — RRF Fusion (`src/hybrid_retriever.py`)
Reciprocal Rank Fusion combines the BM25 and vector ranked lists into one without needing to normalize scores across different scales. The formula is:

```
RRF score = (vector_weight / (k + vector_rank)) + (bm25_weight / (k + bm25_rank))
```

Where `k=60` dampens the influence of top-ranked results and weights are 0.6 / 0.4 in favour of semantic search.

### Step 6 — Cross-Encoder Reranking (`src/reranker.py`)
Takes the top 20 fused candidates and passes each (query, chunk) pair through `ms-marco-MiniLM-L-6-v2`, a cross-encoder model trained specifically on passage relevance. Unlike bi-encoders which embed query and chunk separately, a cross-encoder reads them together — dramatically more accurate. Returns the top 5 chunks by rerank score.

### Step 7 — Generation with Citation Enforcement (`src/generator.py`)
The top 5 chunks are formatted as numbered sources and sent to Groq's `llama-3.3-70b-versatile` with strict instructions: every factual claim must cite `[Source N]`, and if the chunks don't support a confident answer the model must respond with `INSUFFICIENT_EVIDENCE` rather than hallucinate.

### Step 8 — Faithfulness Evaluation (`src/evaluator.py`)
Runs the full pipeline on a golden set of manually verified Q&A pairs. For each answer, an LLM judge reads the answer and the retrieved chunks and scores faithfulness from 0.0 to 1.0. Mean score below 0.70 fails CI.

---

## Evaluation Pipeline

The system includes a CI-gated faithfulness evaluation:

1. **Golden set** — 10 manually verified Q&A pairs in `data/golden_set.json`
2. **LLM-as-judge** — Groq scores each answer for faithfulness to retrieved chunks
3. **CI gate** — GitHub Actions runs eval on every push; build fails if mean faithfulness drops below 0.70

**Latest eval results:**
- Mean faithfulness: **0.7875**
- Pass rate: **100%**
- Status: ✅ CI PASSED

---

## Key Design Decisions

**Why hybrid retrieval?**
Vector search understands meaning but misses exact terms. BM25 matches keywords but misses semantics. Combining both via RRF consistently outperforms either alone — especially on research papers where precise terminology and exact numbers matter.

**Why cross-encoder reranking?**
Bi-encoder embeddings compare query and chunk independently. A cross-encoder reads them together, which is far more accurate at the cost of speed. Running it only on the top 20 candidates (not all chunks) keeps latency acceptable while delivering the accuracy benefit.

**Why citation enforcement?**
The LLM is explicitly instructed to respond with `INSUFFICIENT_EVIDENCE` rather than generate plausible-sounding content when the retrieved chunks don't support an answer. This eliminates hallucination as a failure mode — the system either cites evidence or declines.

**Why a CI evaluation gate?**
Prompt changes, chunking changes, and model changes can silently degrade answer quality. Running automated faithfulness evaluation on every commit catches regressions before they ship. This is how production AI teams operate.

**Why 600 token chunks with 100 token overlap?**
600 tokens gives enough context for an LLM to understand the passage without exceeding practical limits. 100 token overlap ensures sentences that fall at chunk boundaries appear in full in at least one chunk — preventing context loss at boundaries.

---

## CI Setup

Add your `GROQ_API_KEY` to GitHub repository secrets:

**Settings → Secrets and variables → Actions → New repository secret**

Every pull request then automatically:
1. Runs the full RAG pipeline on the golden set
2. Scores faithfulness via LLM-as-judge
3. Fails the build if quality drops below threshold
4. Uploads `eval_report.json` as a downloadable artifact

---

## Adding New Documents

Drop any PDF into `data/raw/` and update the papers list in `src/pipeline.py`:

```python
papers = [
    ("data/raw/attention.pdf",  "Attention Is All You Need"),
    ("data/raw/bert.pdf",       "BERT"),
    ("data/raw/rag_paper.pdf",  "Retrieval-Augmented Generation"),
    ("data/raw/your_paper.pdf", "Your Paper Title"),  # add here
]
```

ChromaDB deduplicates on re-ingestion so running the pipeline again is safe.

---

## Extending the Golden Set

Open `data/golden_set.json` and add entries:

```json
{
    "question": "Your verified question here?",
    "expected_answer": "The correct answer you verified manually.",
    "source_paper": "Paper Title"
}
```

Aim for 50–200 entries in production. Questions should have clear, unambiguous answers directly stated in the documents.

---

## Requirements

```
pypdf
tiktoken
chromadb
sentence-transformers
groq
python-dotenv
pyyaml
loguru
rank-bm25
```

Install: `pip install -r requirements.txt`

---

## Papers Used

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Devlin et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- Lewis et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

## What I Learned Building This

- The gap between a RAG demo and a production RAG system is enormous — most tutorials stop at basic vector search
- Cross-encoder reranking is the single biggest quality improvement in the pipeline and the step most people skip
- Citation enforcement is more important than answer quality — a system that declines to answer is more trustworthy than one that confidently hallucinates
- Automated evaluation is not optional — silent quality regressions are the most dangerous failure mode in AI systems
- Token-aware chunking with overlap is worth the extra complexity — naive character splitting loses too much context at boundaries

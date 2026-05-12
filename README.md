# Medical Specialty Semantic Search

A semantic search and RAG (Retrieval-Augmented Generation) system that maps natural-language medical queries to the most relevant medical specialty. Built as a weekend project to deeply understand the architecture of retrieval-based AI systems end-to-end.

**Example:** the query *"my skin is flaky and itchy"* returns Dermatology with a similarity score of 0.37, followed by Allergy and Immunology (0.31), without either query or any specialty description sharing keywords.

---

## What this project demonstrates

This is a small-scale but architecturally-complete RAG system, built deliberately from primitives (NumPy, OpenAI API) rather than higher-level frameworks (LangChain, LlamaIndex). The goal was to understand every layer of a production RAG pipeline:

1. **Embedding-based retrieval** — embed a corpus of 40 medical specialty descriptions using OpenAI's `text-embedding-3-small`, then rank by cosine similarity to the query embedding.
2. **Grounded generation** — pass the top-k retrieved specialties to `gpt-4o-mini` as context, instructing the model to recommend a specialty *based only on the retrieved information*, or to decline if none fit.
3. **Evaluation** — 30 hand-labeled `(query, expected_specialty)` test cases spanning easy / medium / hard difficulty, with top-1 and top-3 accuracy reported.

**Current results:** 80% top-1 accuracy, 97% top-3 accuracy on the eval set.

---

## Architecture

User query
│
▼
┌─────────────────────────────────────────────────┐
│  Layer 1: Retrieval (src/search.py)             │
│  - Embed query via OpenAI                       │
│  - Cosine similarity vs. 40 cached specialty    │
│    vectors                                       │
│  - Return top-k ranked results                  │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  Layer 2: RAG (src/rag.py)                      │
│  - Inject top-k specialty descriptions into     │
│    LLM prompt                                    │
│  - Low temperature (0.2) for grounded output    │
│  - Instructed to decline if no specialty fits   │
└─────────────────────────────────────────────────┘
│
▼
Grounded natural-language recommendation
Offline (one-time):
data/specialties.json ──▶ src/embed.py ──▶ embeddings.npz
Online (per query):
query ──▶ search.py (or rag.py) ──▶ result

## Technical decisions and rationale

### Why embedding-based retrieval over keyword search?
The hackathon example that inspired this project — mapping *"psoriasis"* to *Dermatology* — only works if the system understands semantic similarity, not lexical overlap. Embedding-based retrieval matches *meanings*; keyword search would miss queries like *"flaky skin"* that share no words with any specialty description.

### Why `text-embedding-3-small` (1536-dim)?
Cost-effective (~$0.02 per million tokens), strong general-domain performance, and outputs L2-normalized vectors — meaning cosine similarity reduces to a dot product, simplifying the math. For a specialized domain at production scale, a domain-specific embedding model could outperform it; for a general-purpose patient-facing query system, this is the right default.

### Why cosine similarity?
Cosine similarity measures the angle between two embedding vectors, which captures semantic alignment regardless of magnitude. Because OpenAI embeddings are unit-normalized, the cosine computation simplifies to `vectors @ query_vector` — a single matrix-vector multiplication that computes all 40 similarities in parallel.

### Why NumPy instead of a vector database?
For 40 vectors, an in-memory NumPy array is faster than any vector DB and has zero infrastructure cost. A vector database (Chroma, pgvector, Pinecone) becomes necessary when (a) the corpus is too large for memory, (b) persistence and transactional consistency matter, or (c) approximate nearest-neighbor indexing is required for latency. The vector DB choice is a function of dataset size, latency requirements, and surrounding infrastructure — not a default.

### Why no framework (LangChain, LlamaIndex)?
The point of building this from scratch was to understand what those frameworks abstract away. Embedding, similarity, top-k selection, and prompt construction are all small, comprehensible operations — knowing them directly makes it possible to evaluate whether a framework adds value for a given problem or just adds dependency surface.

### Why a separate offline indexing script (`embed.py`)?
Embedding the corpus is the expensive operation; queries against pre-computed vectors are cheap. Separating offline indexing from online query is the standard production pattern: re-index when the corpus changes, serve queries from the cached index. This mirrors how any production RAG system is structured.

### Prompt engineering choices for the RAG layer
- **System / user separation:** Instructions live in the system message; the query lives in the user message. This establishes a privilege hierarchy that's important when guarding against prompt injection in production.
- **Explicit decline instruction:** The prompt instructs the model to say "no specialty fits" rather than guess. This is *grounded refusal* — one of the strongest techniques against hallucination in RAG systems.
- **Low temperature (0.2):** Reduces creative drift away from the retrieved context. Higher temperature is appropriate for brainstorming; low temperature is appropriate for factual synthesis.

---

## Evaluation methodology

The eval set (`data/eval_set.json`) contains 30 hand-curated queries with single gold labels, distributed across difficulty levels:

- **Easy (~10):** unambiguous symptom-to-specialty mappings
- **Medium (~15):** paraphrased or layperson-language queries
- **Hard (~5):** genuinely ambiguous queries where multiple specialties are clinically defensible

Two metrics are tracked:
- **Top-1 accuracy** — how often the single best match is the gold label
- **Top-3 accuracy** — how often the gold label appears in the top 3 results

The gap between top-1 and top-3 also signals query ambiguity: a small gap means most queries have a clear single best answer, a large gap signals that retrieval is finding the right answer but not always ranking it first.

### Failure analysis (the part worth reading)

Of the two queries that failed top-3 evaluation, the analysis revealed two distinct failure types worth distinguishing:

**Failure type 1: corpus quality (fixable).** A query about jaundice (*"my eyes look yellow and my skin is itchy all over"*) failed to retrieve Hepatology, because the original Hepatology description used the clinical term *"jaundice"* but not the layperson terms *"yellow skin"* or *"yellow eyes."* The fix was to enrich the description with patient-presenting language and re-embed. This is the **vocabulary gap** problem — embeddings match on the text they were given, so the text must reflect both expert and lay vocabulary.

**Failure type 2: irreducible ambiguity (kept as a known limitation).** A query about dizziness (*"the room spins for a few seconds"*) retrieved Neurology, Cardiology, and Sleep Medicine, all clinically reasonable, while we had labeled Otolaryngology as the gold answer. "Fixing" this would amount to over-fitting the embeddings to the eval set; in reality, dizziness is multi-causal and a real medical system would route to multiple specialties or ask follow-up questions. This failure is preserved deliberately to illustrate the limits of single-gold-label evaluation.

### What the eval methodology cannot capture

- **Multiple correct answers** — single gold labels unfairly punish multi-specialty queries
- **Severity awareness** — the system doesn't know that chest pain is urgent and rashes are not
- **Patient history / follow-up** — single-shot retrieval can't replicate a clinician's interactive triage

---

## Out-of-domain behavior (grounded refusal)

A key behavior demonstrated in the RAG layer: when a query is unrelated to medicine (e.g., *"I have to file my taxes"*), the retrieval scores drop below ~0.15 across all specialties, and the LLM correctly responds *"None of the listed specialties are clearly appropriate."* This is enabled by two combined techniques: low similarity scores triggering an absence of strong signal in the prompt, plus an explicit "decline if nothing fits" instruction. **Grounded refusal is what separates a real RAG system from a chatbot that hallucinates confidently.**

---

## Project structure
medical-specialty-search/
├── data/
│   ├── specialties.json    # 40 specialty names + descriptions
│   └── eval_set.json       # 30 labeled test cases
├── src/
│   ├── embed.py            # one-time: build embeddings.npz
│   ├── search.py           # retrieval layer (Layer 1)
│   └── rag.py              # generation layer (Layer 2)
│   └── evaluation.py         # accuracy script (Layer 3)
├── embeddings.npz          # generated; gitignored
├── .env                    # OPENAI_API_KEY; gitignored
└── requirements.txt

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
echo OPENAI_API_KEY=sk-... > .env

python src/embed.py             # one-time, generates embeddings.npz
python src/search.py            # interactive retrieval CLI
python src/rag.py               # interactive RAG CLI
python src/evaluation.py        # run the eval harness
```

---

## What would change at production scale

This project is small-by-design. At a production scale (e.g., embedding tens of thousands of enterprise documents for use cases like project-cost estimation, spec validation, or compliance checking), several decisions would shift:

- **Vector store:** in-memory NumPy → pgvector (if persistence + relational data is needed) or a managed ANN service (Pinecone, Qdrant) for very large corpora
- **Embedding cost:** batching with rate-limit handling and resumable jobs; possibly an embedding cache keyed by content hash
- **Retrieval quality:** hybrid search (combining BM25 keyword + embedding scores) to handle exact-match needs (part numbers, project IDs); optional reranking with a cross-encoder for top results
- **Evaluation:** continuous evaluation against a growing, versioned eval set; regression testing on every embedding-model or prompt change
- **Observability:** logging of query, retrieved IDs, scores, and final answer for offline analysis and red-teaming
- **Prompt injection defense:** retrieved content is untrusted; the system prompt must constrain the model's role even when context is adversarial

---

## Stack

- Python 3.x
- OpenAI API (`text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation)
- NumPy for vector math and storage
- python-dotenv for secrets management
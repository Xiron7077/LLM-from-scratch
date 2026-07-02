# Custom-Transformers-and-ArXiv-RAG

Two related NLP projects in this repo: custom Transformer architectures built from scratch in PyTorch, and a fully local Retrieval-Augmented Generation (RAG) pipeline for querying ArXiv papers.

## Project 1: Transformers From Scratch (`Transformers.ipynb`)

- Implemented **encoder-only**, **decoder-only**, and **encoder-decoder** Transformer architectures from low-level components in PyTorch - no high-level Transformer library shortcuts.
- Core mechanisms built from scratch: multi-head self-attention, positional encoding, layer normalization, masked self-attention for autoregressive decoding, and cross-attention.
- Architecture: 4 layers, 4 attention heads, d_model = 256, feed-forward dim = 256.
- Trained on 1,992 scientific paper abstracts; vocabulary built from 3,984 texts, vocab size 8,000 tokens.
- Training setup: batch size 32, learning rate 0.0003, 50 epochs.
- Validated across three tasks: abstract classification, autoregressive text generation, and summarization.

## Project 2: Local RAG Pipeline for ArXiv Papers (`RAG.ipynb`)

A fully local retrieval-augmented generation pipeline - no external API calls for generation.

**Data ingestion**
- Downloads 3 seminal papers directly from ArXiv: *Attention Is All You Need*, *BERT*, and *LoRA*.
- Pulls up to 25 additional recent papers from ArXiv (query: `cat:cs.CL AND (LLM OR Transformer)`, sorted by submission date). Actual count may vary run-to-run depending on successful downloads.

**Ingestion & indexing**
- Text extraction via PyMuPDF (`fitz`), page by page.
- Chunking: fixed-size word chunks (250 words, 30-word overlap), chunks under 50 words discarded, each chunk tagged with `[filename, page]` metadata.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Vector index: FAISS (`IndexFlatL2`).

**Generation**
- Local `Qwen/Qwen2.5-1.5B-Instruct` model via Hugging Face Transformers (float16 on GPU, float32 on CPU).
- System prompt enforces per-claim source citation (e.g. `[1]`, `[2]`) grounded strictly in retrieved context.
- Two modes: `generate_vanilla` (no context, model's own knowledge) and `generate_rag` (context-grounded), plus a "Related Work"-style summarization mode over multiple retrieved chunks.

**Evaluation harness**
Runs a side-by-side vanilla-vs-RAG comparison across five question types designed to probe different failure modes:
1. Factual recall (BERT's pre-training tasks)
2. Specific detail retrieval (LoRA's typical rank `r`)
3. Definition accuracy (Scaled Dot-Product Attention)
4. Synthesis across multiple papers (recent efficient-training methods)
5. Negative constraint / hallucination check (a topic absent from the corpus, to see if the model correctly says "not found" instead of making something up)

## Setup

```bash
pip install -U bitsandbytes transformers accelerate sentence-transformers pymupdf faiss-cpu arxiv hf_xet
```

## Repository Structure

```
├── Transformers.ipynb      # Custom Transformer architectures
├── RAG.ipynb               # Local RAG pipeline for ArXiv papers
├── scitldr_train.jsonl     # Training data (SciTLDR summarization dataset)
├── requirements.txt
├── Reports/                # Generated reports / evaluation outputs
└── README.md
```
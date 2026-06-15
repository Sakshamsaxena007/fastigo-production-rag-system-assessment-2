# API Documentation

## Ingestion

### ingest_all()

Loads:

- PDF files
- HTML files
- CSV files

Creates:

- Chunks
- Embeddings
- ChromaDB storage

---

## Retrieval

### retrieve_docs(query)

Input:

- Query string

Output:

- Relevant document chunks

---

## Generation

### Ask(question)

Input:

- User question

Output:

- Generated answer
- Source documents

Features:

- Hallucination control
- Prompt injection defense
- Session memory
- Fallback mechanism

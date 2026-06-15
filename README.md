# Production-Grade RAG Assistant

## Overview

This project is a production-grade Retrieval-Augmented Generation (RAG) assistant developed for the Fastigo AI Engineer Assessment.

The system supports ingestion of PDF, HTML, and CSV documents, stores embeddings in ChromaDB, retrieves relevant context, and generates grounded responses using Groq LLMs.

---

## Features

### Data Ingestion

* PDF Support
* HTML Support
* CSV Support
* Recursive Chunking Strategy

### Retrieval Pipeline

* Sentence Transformer Embeddings
* Chroma Vector Database
* Similarity Search
* Context Compression

### Generation Layer

* Groq LLM Integration
* Source Citations
* Session Memory

### Reliability & Security

* Prompt Injection Defense
* Hallucination Detection
* Fallback Model Strategy

---

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API

---

## Project Structure

data/
docs/
app.py
generator.py
retriever.py
ingest.py

---

## Installation

pip install -r requirements.txt

---

## Run

streamlit run app.py

---

## Present By

Saksham Saxena

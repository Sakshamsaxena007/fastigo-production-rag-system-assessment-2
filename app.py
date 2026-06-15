"""
Main Application - Production-Grade RAG Assessment-2
Author: Saksham Saxena
"""

import streamlit as st
import os
from pathlib import Path
from generator import ask
from ingest import ingest_all

# Page config
st.set_page_config(
    page_title="Production RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Production-Grade RAG Assistant")
st.caption("Upload documents and ask questions — powered by Groq + ChromaDB")

# Sidebar — file upload
with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, HTML or CSV files",
        type=["pdf", "html", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            ext = file.name.split(".")[-1]
            save_path = Path(f"data/{ext}/{file.name}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

    if st.button("⚙️ Process & Ingest Documents"):
        with st.spinner("Ingesting documents into ChromaDB..."):
            ingest_all()
        st.success("✅ Documents ingested successfully!")

    st.divider()
    st.caption("Built for Fastigo AI Engineer Assessment")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(question)
            answer = result["answer"]
            sources = result["source_documents"]

            st.markdown(answer)

            # Show sources
            if sources:
                with st.expander("📄 Sources"):
                    for doc in sources:
                        src = doc.metadata.get("source", "Unknown")
                        st.write(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer})


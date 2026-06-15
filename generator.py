import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Simple in-memory session history
chat_history = []

def get_llm(primary=True):
    model = (
        "llama-3.1-8b-instant"
        if primary
        else "llama3-70b-8192"
    )
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model,
        temperature=0.2
    )

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def ask(question: str):
    banned = [
    "ignore previous",
    "ignore all previous",
    "forget instructions",
    "reveal hidden",
    "secret information",
    "you are now",
    "disregard",
    "override instructions"
    ]
    if any(phrase in question.lower() for phrase in banned):
        return {
            "answer": "⚠️ Invalid query detected. Please ask a genuine question.",
            "source_documents": []
        }

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        print("\nRetrieved Documents:")
        for i, doc in enumerate(docs):
            print(f"\n--- Doc {i+1} ---")
            print(doc.page_content[:500])

        context = "\n\n".join([doc.page_content for doc in docs])
        print("\n========== RETRIEVED CONTEXT ==========")
        print(context)
        print("======================================\n")
        history_text = "\n".join([
            f"Human: {m['human']}\nAI: {m['ai']}"
            for m in chat_history[-3:]
        ])

        prompt = f"""
Answer the user's question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        try:
            llm = get_llm(primary=True)
            response = llm.invoke(prompt)

        except Exception:
            print("Primary model failed. Switching to fallback model...")

            llm = get_llm(primary=False)
            response = llm.invoke(prompt)

        answer = response.content
    
        # Save to history
        chat_history.append({"human": question, "ai": answer})

        return {"answer": answer, "source_documents": docs}

    except Exception as e:
        return {
            "answer": f"⚠️ Error: {str(e)}",
            "source_documents": []
        }

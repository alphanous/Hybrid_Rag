# Local RAG Pipeline with Ollama & Multi-User Caching

A production-ready, fully local Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **LangChain**, and **Ollama**.

This project features intelligent query rewriting, hybrid retrieval (BM25 + Vector), and a secure multi-user caching system that runs entirely on your local machine.

## 🚀 Key Features

* **100% Local Privacy:** Uses Llama 3.2 and Nomic Embeddings via Ollama. No data leaves your machine.
* **Multi-User Caching:** Isolates cache data per user (e.g., `alice_cache.json`, `bob_cache.json`). Alice's previous answers speed up her future queries but remain invisible to Bob.
* **Intelligent Query Rewriting:** Automatically transforms vague follow-up questions (e.g., "How much is it?") into standalone search queries based on chat history.
* **Hybrid Retrieval:** Combines keyword search (BM25) with semantic search (ChromaDB) using Reciprocal Rank Fusion (RRF) for higher accuracy.
* **Stateful Memory:** Maintains conversation history within a session using LangGraph's persistent state.

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.9+**
2.  **Ollama**: Download from [ollama.com](https://ollama.com).

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alphanous/Hybrid_Rag
    cd Hybrid_Rag
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install langchain-ollama langgraph langchain-community chromadb rank_bm25 numpy
    ```

3.  **Pull required Ollama models:**
    You must have the LLM and Embedding models downloaded locally.
    ```bash
    ollama pull llama3.1:latest
    ollama pull nomic-embed-text
    ```

## 📂 Project Structure

```text
├── data/                  # Place your .txt source documents here
├── user_caches/           # Auto-generated folder for user-specific JSON caches
├── chroma_db_ollama/      # Auto-generated Vector Database storage
├── rag_app.py             # Main application script
└── README.md              # This file

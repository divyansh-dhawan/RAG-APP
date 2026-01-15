# RAG Chatbot with Qdrant, Gemini, and Cohere

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-app-ohyrmcnjhvjr7axbshypvk.streamlit.app/)

This project is a robust RAG (Retrieval-Augmented Generation) chatbot that uses Qdrant as a vector database, Gemini for embeddings and language model, and Cohere for reranking. Built with Streamlit for a clean UI.

## Features

*   **Smart Rate Limiting**: Automatically handles Google API's `429 Resource Exhausted` errors by parsing dynamic retry times.
*   **Model Fallback Strategy**: "Self-healing" generation that rotates between `gemini-2.5-flash`, `gemini-2.5-flash-lite`, and `gemini-3-flash-preview` to maximise uptime.
*   **Modern Stack**: Uses `langchain-qdrant` for Qdrant connectivity and `gemini-2.5/3.0` models.

## Architecture

1.  **Data Ingestion**: Upload text files or paste content.
2.  **Chunking**: RecursiveCharacterTextSplitter (Size: 1000, Overlap: 100).
3.  **Embedding**: `models/text-embedding-004` (Google Gemini).
4.  **Vector Store**: Qdrant Cloud (via `langchain-qdrant`).
5.  **Reranking**: Cohere `rerank-english-v3.0`.
6.  **Answering**: Gemini 2.5/3.0 generation with source citations.

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/divyansh-dhawan/RAG-APP.git
    cd rag-app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Secrets:**
    The app uses Streamlit secrets. Create a file at `.streamlit/secrets.toml`:
    ```toml
    GOOGLE_API_KEY = "your-google-key"
    COHERE_API_KEY = "your-cohere-key"
    QDRANT_API_KEY = "your-qdrant-key"
    QDRANT_URL = "your-qdrant-url"
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## Providers

*   **Vector Database**: [Qdrant Cloud](https://qdrant.tech/cloud/)
*   **Embeddings**: Google Gemini `text-embedding-004`
*   **LLM**: Google Gemini `2.5-flash` / `3-flash-preview`
*   **Reranker**: Cohere `rerank-english-v3.0`

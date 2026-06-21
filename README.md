# RAG Chatbot with Qdrant, Gemini, and Cohere

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-app-ohyrmcnjhvjr7axbshypvk.streamlit.app/)

This project is a robust RAG (Retrieval-Augmented Generation) chatbot that uses Qdrant as a vector database, Gemini for embeddings and language model, and Cohere for reranking. Built with Streamlit for a clean UI.

## Features

*   **Smart Rate Limiting**: Automatically handles Google API's `429 Resource Exhausted` errors by parsing dynamic retry times.
*   **Model Fallback Strategy**: "Self-healing" generation that rotates between `gemini-2.5-flash`, `gemini-2.5-flash-lite`, and `gemini-3-flash-preview` to maximise uptime.
*   **Modern Stack**: Uses `langchain-qdrant` for Qdrant connectivity and `gemini-2.5/3.0` models.
*   **PDF Upload & Parsing**: Upload text-based PDF files directly through the Streamlit UI — the app extracts text, chunks it, creates embeddings, and stores them in Qdrant for retrieval.

## Architecture

1.  **Data Ingestion**: Upload text files, paste content, or upload a PDF (text-based). Scanned/image PDFs may require OCR and are not guaranteed to be parsed correctly without additional tooling.
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

5.  **Using the PDF upload feature:**
    - Open the Streamlit app in your browser.
    - Use the Upload area to select a PDF file (text-based PDFs are supported).
    - The app will extract the PDF's text, chunk it, and add it to the vector store for retrieval and question-answering.
    - If your PDF is a scanned document (images), consider running OCR before uploading or convert it to a machine-readable PDF.

## Providers

*   **Vector Database**: [Qdrant Cloud](https://qdrant.tech/cloud/)
*   **Embeddings**: Google Gemini `text-embedding-004`
*   **LLM**: Google Gemini `2.5-flash` / `3-flash-preview`
*   **Reranker**: Cohere `rerank-english-v3.0`

## Notes & Tips

*   The PDF upload currently supports text-based PDFs. OCR workflows (for image PDFs) are out of scope for the built-in parser — if you need OCR, integrate tools like Tesseract or commercial OCR services before ingestion.
*   Make sure your `.streamlit/secrets.toml` is not checked into source control — it contains API keys.


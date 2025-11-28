# RAG Chatbot with Qdrant, Gemini, and Cohere

This project is a simple RAG (Retrieval-Augmented Generation) chatbot that uses Qdrant as a vector database, Gemini for embeddings and language model, and Cohere for reranking. The application is built with Streamlit.

## Architecture

1.  **Data Ingestion**: Users can upload a text file or paste text into a text area.
2.  **Chunking**: The input text is split into smaller chunks.
3.  **Embedding**: The text chunks are converted into vector embeddings using Google's Gemini model.
4.  **Vector Store**: The embeddings are stored in a Qdrant Cloud collection.
5.  **Retrieval**: When a user asks a question, the application retrieves the most relevant documents from Qdrant.
6.  **Reranking**: The retrieved documents are reranked using Cohere's Rerank API to improve relevance.
7.  **Answering**: The reranked documents and the user's question are passed to the Gemini 1.5 Flash model to generate a final answer.
8.  **Citations**: The answer is presented with citations to the source documents.

## Chunking Parameters

*   **Chunk Size**: 1000 tokens
*   **Chunk Overlap**: 100 tokens (10%)

## Retriever and Reranker Settings

*   **Retriever**:
    *   **Top-k**: 5
*   **Reranker**:
    *   **Model**: `rerank-english-v2.0`
    *   **Top-n**: 3

## Providers Used

*   **Vector Database**: [Qdrant Cloud](https://qdrant.tech/cloud/)
*   **Embedding Model**: [Google Gemini](https://ai.google.dev/) (`models/embedding-001`)
*   **LLM**: [Google Gemini 1.5 Flash](https://ai.google.dev/)
*   **Reranker**: [Cohere](https://cohere.com/)

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add your API keys:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    COHERE_API_KEY="YOUR_COHERE_API_KEY"
    QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
    QDRANT_URL="YOUR_QDRANT_URL"
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Remarks

*   The cost estimation in the app is a rough estimate based on the number of tokens. For accurate pricing, please refer to the official pricing pages of the respective providers.
*   The current implementation uses a simple chunking strategy. For more complex documents, more advanced chunking strategies might be needed.
*   The application is designed for demonstration purposes and may have limitations in a production environment.

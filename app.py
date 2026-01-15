import streamlit as st
import os
import time
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import qdrant_client
from dotenv import load_dotenv
import cohere

# --- Helper Functions ---

# Load environment variables
load_dotenv()

# Securely load secrets with error handling
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
except Exception as e:
    st.error(f"Missing API Keys in secrets.toml: {e}")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Data Ingestion and Chunking
def process_and_chunk_data(text_data, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text_data)
    metadatas = [{"source": f"chunk-{i}", "title": "User Uploaded Data", "position": i} for i in range(len(docs))]
    return docs, metadatas

# Vector Store (Qdrant)
def get_qdrant_vector_store(collection_name="user-data-collection"):
    # print(f"[DEBUG] Connecting to Qdrant at {QDRANT_URL}")
    client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    # Check if collection exists, if not create it
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        # print(f"[DEBUG] Collection '{collection_name}' does not exist. Attempting to create...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_client.http.models.VectorParams(size=768, distance=qdrant_client.http.models.Distance.COSINE)
            )
        except Exception as e_create:
            st.error(f"Failed to create collection: {repr(e_create)}")
            raise RuntimeError(f"Failed to create collection '{collection_name}': {str(e_create)}") from e_create

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store

def upsert_data_to_qdrant(documents, metadatas, collection_name="user-data-collection", batch_size=1):
    """
    Upserts documents and metadatas to the Qdrant collection in batches.
    Includes rate limiting to avoid Google API 429 errors.
    """
    vector_store = get_qdrant_vector_store(collection_name)
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0, text="Embedding data (Slow mode for Free Tier)...")
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        current_batch = i // batch_size + 1
        
        # Simple retry logic for 429 errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Capture IDs to verify insertion
                ids = vector_store.add_texts(batch_docs, batch_metadatas)
                st.write(f"DEBUG: Batch {current_batch} returned {len(ids)} IDs") # Debug
                break
            except Exception as e:
                # If last attempt, raise
                if attempt == max_retries - 1:
                    raise e
                    
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    # Exponential backoff: 30s, 60s
                    wait_time = (attempt + 1) * 30 
                    st.toast(f"Rate limit hit. Waiting {wait_time}s...", icon="⏳")
                    time.sleep(wait_time)
                else:
                    raise e
        
        # Update progress
        progress = min(current_batch / total_batches, 1.0)
        progress_bar.progress(progress, text=f"Embedding batch {current_batch}/{total_batches}...")
        
        # Forced throttling for free tier (15 RPM limit -> 1 req every 4s)
        time.sleep(5) 
        
    progress_bar.empty()
    # print(f"Upserted {len(documents)} chunks to Qdrant collection '{collection_name}'.")

# Retriever and Reranker
def get_retriever(collection_name="user-data-collection", top_k=5):
    vector_store = get_qdrant_vector_store(collection_name)
    return vector_store.as_retriever(search_kwargs={"k": top_k})

def rerank_documents(query, documents, top_n=3):
    if not documents:
        return []
    try:
        co = cohere.Client(COHERE_API_KEY)
        docs_content = [doc.page_content for doc in documents]
        results = co.rerank(query=query, documents=docs_content, top_n=top_n, model="rerank-english-v3.0")
        return [documents[result.index] for result in results.results]
    except Exception as e:
        st.warning(f"Reranking failed (using original retrieval): {e}")
        return documents[:top_n]

def get_vector_count(collection_name="user-data-collection"):
    """
    Returns the number of vectors in the Qdrant collection.
    """
    client = qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        count = client.count(collection_name=collection_name, exact=True).count
    except Exception as e:
        st.error(f"Error counting vectors: {e}")
        count = 0
    return count

def delete_all_points_from_collection(collection_name="user-data-collection"):
    """
    Deletes all points from the Qdrant collection without deleting the collection.
    """
    client = qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_client.http.models.FilterSelector(
                filter=qdrant_client.http.models.Filter(
                    must=[],
                )
            )
        )
    except Exception as e:
        print(f"Could not delete points: {e}")

# --- Streamlit App ---

st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="🤖")

def inject_custom_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_custom_css()

st.markdown('<div class="main-header">RAG Chatbot Enterprise</div>', unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("Data Ingestion")
    st.markdown("Upload a text file or paste content directly to add to the knowledge base.")
    
    tab1, tab2 = st.tabs(["📁 File Upload", "📝 Text Input"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    
    with tab2:
        text_input = st.text_area("Paste text here", height=150)
    
    process_btn = st.button("🚀 Process and Store Data", use_container_width=True)
    
    if process_btn:
        if uploaded_file is not None:
            text_data = uploaded_file.read().decode("utf-8")
            st.toast("Processing uploaded file...", icon="⏳")
        elif text_input.strip():
            text_data = text_input
            st.toast("Processing pasted text...", icon="⏳")
        else:
            st.warning("⚠️ Please upload a file or paste text to process.")
            st.stop()

        with st.spinner("Chunking and upserting data to Qdrant..."):
            start_time = time.time()
            try:
                documents, metadatas = process_and_chunk_data(text_data)
                st.write(f"DEBUG: Generated {len(documents)} chunks.") # Debug
                
                upsert_data_to_qdrant(documents, metadatas)
                
                # Check vector count immediately
                vector_count = get_vector_count()
                st.write(f"DEBUG: Vector count after upsert: {vector_count}") # Debug
                
                end_time = time.time()
                st.success(f"✅ Successfully stored {len(documents)} chunks in {end_time - start_time:.2f}s.")
                st.session_state.data_processed = True
                time.sleep(2)
                st.rerun()
            except Exception as e:
                import traceback
                st.error(f"❌ Error: {str(e)}")
                # st.write(f"Detailed Error: {e}")
                traceback.print_exc()

    st.divider()
    st.header("📊 Database Status")
    vector_count = get_vector_count()
    st.metric("Total Vectors", vector_count)

    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        with st.spinner("Cleaning up..."):
            delete_all_points_from_collection()
            st.success("Database cleared!")
            time.sleep(1)
            st.rerun()

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

# Initial Welcome Message
if not st.session_state.messages and vector_count == 0:
    st.info("👋 Welcome! Please convert some text into vectors using the sidebar to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your data..."):
    if vector_count == 0:
        st.warning("⚠️ Please process data before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                try:
                    # Retrieval
                    retriever = get_retriever()
                    retrieved_docs = retriever.invoke(query)
                    
                    # Reranking
                    reranked_docs = rerank_documents(query, retrieved_docs)
                    
                    # Answering
                    context = "\n\n".join([doc.page_content for doc in reranked_docs])
                    
                    prompt_template = """
                    Answer the question as detailed as possible from the provided context. If the answer is not in the
                    provided context just say, "answer is not available in the context", don't provide a wrong answer.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """
                    
                    # Model Fallback Strategy
                    # Priority: 2.5 Flash (Stable) -> 2.5 Lite (Fast) -> 3.0 Preview (New) -> 2.5 Pro (Powerful but strict limits)
                    candidate_models = [
                        "gemini-2.5-flash", 
                        "gemini-2.5-flash-lite", 
                        "gemini-3-flash-preview",
                        "gemini-2.5-pro"
                    ]
                    answer = ""
                    last_exception = None
                    
                    for model_name in candidate_models:
                        try:
                            # st.caption(f"Trying model: `{model_name}`...")
                            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, api_key=GOOGLE_API_KEY)
                            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                            chain = prompt | model
                            
                            # Retry logic (per model)
                            max_retries = 3
                            success = False
                            
                            for attempt in range(max_retries):
                                try:
                                    response = chain.invoke({"context": context, "question": query})
                                    answer = response.content
                                    success = True
                                    break # Inner retry loop
                                except Exception as e:
                                    last_exception = e
                                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                        if attempt < max_retries - 1:
                                            import re
                                            wait_time = 20 # Default
                                            retry_match = re.search(r"retry in (\d+\.?\d*)s", str(e))
                                            if retry_match:
                                                wait_time = float(retry_match.group(1)) + 5
                                            
                                            st.toast(f"Rate limit on {model_name}. Waiting {wait_time:.1f}s...", icon="⏳")
                                            time.sleep(wait_time)
                                            continue
                                    # If not 429, or retries exhausted, re-raise to switch model
                                    raise e
                            
                            if success:
                                break # Outer model loop (Success!)
                                
                        except Exception as e:
                            st.warning(f"Model `{model_name}` failed. Switching to next backup...")
                            last_exception = e
                            continue
                            
                    if not answer and last_exception:
                        raise last_exception

                    # Create citations
                    citations = [f"**[{i+1}]** {doc.metadata.get('title', 'Unknown Title')} (chunk {doc.metadata.get('position', 'N/A')})" for i, doc in enumerate(reranked_docs)]
                    
                    end_time = time.time()
                    
                    full_response = f"{answer}\n\n---\n**📚 Sources:**\n" + "\n".join(citations)
                    message_placeholder.markdown(full_response)
                    
                    st.caption(f"⏱️ Response generated in {end_time - start_time:.2f} seconds.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    full_response = "I encountered an error while processing your request."
                    message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        if 'reranked_docs' in locals() and reranked_docs:
            with st.expander("🔍 Show Detailed Reranked Sources"):
                for i, doc in enumerate(reranked_docs):
                    st.markdown(f"#### Source [{i+1}]")
                    st.caption(f"**Title:** {doc.metadata.get('title', 'N/A')} | **Chunk:** {doc.metadata.get('position', 'N/A')}")
                    st.info(doc.page_content)
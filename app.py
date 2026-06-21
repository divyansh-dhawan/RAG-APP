import streamlit as st
import langchain_google_genai
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
except Exception as e:
    st.error(f"Missing API Keys: {e}")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

try:
    models = list(genai.list_models())

    st.write("Models found:", len(models))

    for m in models:
        st.write(m.name)

except Exception as e:
    st.exception(e)
'''
import time
import re
import cohere
import qdrant_client
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="🤖")

# Load CSS
def inject_custom_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Graceful fallback
inject_custom_css()

# Load Secrets
load_dotenv()
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
except Exception as e:
    st.error(f"Missing API Keys: {e}")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ==============================================================================
# STEP 1: DATA INGESTION (Processing & Chunking)
# ==============================================================================
# In this step, we take the raw text and break it down into smaller, manageable
# pieces ("chunks") so they can be embedded and stored efficiently.
# ==============================================================================
def process_and_chunk_data(text_data, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text_data)
    metadatas = [{"source": f"chunk-{i}", "title": "User Uploaded Data", "position": i} for i in range(len(docs))]
    return docs, metadatas

# ==============================================================================
# STEP 2: VECTOR STORE (Storage & Indexing)
# ==============================================================================
# Here we connect to the Qdrant database. We take the text chunks, turn them into
# numerical vectors (embeddings) using Google's model, and save them.
# ==============================================================================
def get_qdrant_vector_store(collection_name="user-data-collection"):
    client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-001", google_api_key=GOOGLE_API_KEY)

    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=768, 
                    distance=qdrant_client.http.models.Distance.COSINE
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e}")

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

def upsert_data_to_qdrant(documents, metadatas, collection_name="user-data-collection", batch_size=1):
    vector_store = get_qdrant_vector_store(collection_name)
    total_batches = (len(documents) + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="Embedding data ...")
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        current_batch = i // batch_size + 1
        
        # Retry logic for Rate Limits
        for attempt in range(3):
            try:
                vector_store.add_texts(batch_docs, batch_meta)
                break
            except Exception as e:
                if attempt == 2: raise e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep((attempt + 1) * 30)
                else:
                    raise e
        
        progress_bar.progress(min(current_batch / total_batches, 1.0))
        time.sleep(5) # Throttle for Free Tier
    progress_bar.empty()

def get_vector_count(collection_name="user-data-collection"):
    client = qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        return client.count(collection_name=collection_name, exact=True).count
    except:
        return 0

def clean_database(collection_name="user-data-collection"):
    client = qdrant_client.QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    client.delete(collection_name=collection_name, points_selector=qdrant_client.http.models.FilterSelector(filter=qdrant_client.http.models.Filter(must=[])))

# ==============================================================================
# STEP 3: RETRIEVAL & RERANKING (Finding the Best Context)
# ==============================================================================
# First, we find the roughly similar chunks from Qdrant (Retrieval).
# Then, we use Cohere to re-order them so the MOST relevant one is truly #1 (Reranking).
# ==============================================================================
def get_retriever(collection_name="user-data-collection", top_k=5):
    vector_store = get_qdrant_vector_store(collection_name)
    return vector_store.as_retriever(search_kwargs={"k": top_k})

def rerank_documents(query, documents, top_n=3):
    if not documents: return []
    try:
        co = cohere.Client(COHERE_API_KEY)
        docs_content = [doc.page_content for doc in documents]
        results = co.rerank(query=query, documents=docs_content, top_n=top_n, model="rerank-english-v3.0")
        return [documents[result.index] for result in results.results]
    except Exception as e:
        st.warning(f"Reranking failed: {e}")
        return documents[:top_n]

# ==============================================================================
# STEP 4: GENERATION (The AI Answer)
# ==============================================================================
# We feed the User Query + Reranked Context to Geminim, using a smart
# fallback strategy to ensure we always get an answer despite API limits.
# ==============================================================================
def generate_answer(query, context):
    prompt_template = """
    Answer the question detailedly based ONLY on the provided context.
    If the answer is not in the context, say "answer is not available".
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    candidate_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-2.5-pro"]
    
    for model_name in candidate_models:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, api_key=GOOGLE_API_KEY)
            chain = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) | model
            
            # Robust Retry Loop
            for attempt in range(3):
                try:
                    return chain.invoke({"context": context, "question": query}).content
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < 2:
                            wait = 20
                            match = re.search(r"retry in (\d+\.?\d*)s", str(e))
                            if match: wait = float(match.group(1)) + 5
                            st.toast(f"Rate limit on {model_name}. Waiting {wait:.1f}s...", icon="⏳")
                            time.sleep(wait)
                            continue
                    raise e
        except Exception:
            st.warning(f"Model {model_name} busy. Switching...")
            continue
            
    raise RuntimeError("All models failed.")

# ==============================================================================
# UI INTERFACE
# ==============================================================================
st.markdown('<div class="main-header">RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Data Ingestion")
    tab1, tab2 = st.tabs(["📁 File Upload", "📝 Text Input"])
    with tab1: file = st.file_uploader("Upload text", type=["txt"])
    with tab2: text = st.text_area("Paste text", height=150)
    
    if st.button("🚀 Process Data", use_container_width=True):
        raw_text = file.read().decode("utf-8") if file else text
        if raw_text.strip():
            with st.spinner("Processing..."):
                docs, meta = process_and_chunk_data(raw_text)
                upsert_data_to_qdrant(docs, meta)
                st.success("Data stored!")
                time.sleep(1)
                st.rerun()
                
    st.divider()
    curr_count = get_vector_count()
    st.metric("Total Vectors", curr_count)
    if st.button("🗑️ Clear DB", use_container_width=True):
        clean_database()
        st.rerun()

# Chat
if "messages" not in st.session_state: st.session_state.messages = []
if curr_count == 0: st.info("👋 Upload data to start!")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if query := st.chat_input("Ask a question..."):
    if curr_count == 0: st.stop()
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = get_retriever().invoke(query)
            reranked = rerank_documents(query, retrieved)
            context_str = "\n\n".join([d.page_content for d in reranked])
            
            answer = generate_answer(query, context_str)
            
            sources = [f"**[{i+1}]** {d.metadata.get('title','Doc')} (chunk {d.metadata.get('position','?')})" for i,d in enumerate(reranked)]
            full_resp = f"{answer}\n\n---\n**📚 Sources:**\n" + "\n".join(sources)
            st.markdown(full_resp)
            
    st.session_state.messages.append({"role": "assistant", "content": full_resp})
'''

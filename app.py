import streamlit as st
import os
import time
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import qdrant_client
from dotenv import load_dotenv
import cohere
import streamlit as st
# --- Helper Functions ---

# Load environment variables
load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
cohere_api_key = st.secrets["COHERE_API_KEY"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]

# Data Ingestion and Chunking
def process_and_chunk_data(text_data, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text_data)
    metadatas = [{"source": f"chunk-{i}", "title": "User Uploaded Data", "position": i} for i in range(len(docs))]
    return docs, metadatas

# Vector Store (Qdrant)
def get_qdrant_vector_store(collection_name="user-data-collection"):
    print(f"[DEBUG] Connecting to Qdrant at {qdrant_url}")
    client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if collection exists, if not create it (safer than recreate_collection)
    try:
        client.get_collection(collection_name=collection_name)
        print(f"[DEBUG] Collection '{collection_name}' exists.")
    except Exception as e_get:
        # If collection does not exist, create it (does not delete anything)
        print(f"[DEBUG] Collection '{collection_name}' does not exist. Attempting to create...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_client.http.models.VectorParams(size=768, distance=qdrant_client.http.models.Distance.COSINE)
            )
            print(f"[DEBUG] Collection '{collection_name}' created successfully.")
        except Exception as e_create:
            print(f"[DEBUG] Failed to create collection: {repr(e_create)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create collection '{collection_name}': {str(e_create)}") from e_create

    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vector_store

def upsert_data_to_qdrant(documents, metadatas, collection_name="user-data-collection", batch_size=32):
    """
    Upserts documents and metadatas to the Qdrant collection in batches.
    """
    vector_store = get_qdrant_vector_store(collection_name)
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        vector_store.add_texts(batch_docs, batch_metadatas)
    print(f"Upserted {len(documents)} chunks to Qdrant collection '{collection_name}'.")

# Retriever and Reranker
def get_retriever(collection_name="user-data-collection", top_k=5):
    vector_store = get_qdrant_vector_store(collection_name)
    return vector_store.as_retriever(search_kwargs={"k": top_k})

def rerank_documents(query, documents, top_n=3):
    co = cohere.Client(cohere_api_key)
    docs = [doc.page_content for doc in documents]
    results = co.rerank(query=query, documents=docs, top_n=top_n, model="rerank-english-v3.0")
    return [documents[result.index] for result in results.results]

# LLM and Answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=get_retriever(), chain_type_kwargs={"prompt": prompt}, return_source_documents=True)
    return chain

def generate_response(query, chain):
    result = chain({"query": query})
    answer = result["result"]
    source_documents = result["source_documents"]
    
    citations = [f"[{i+1}] {doc.metadata.get('title', 'Unknown Title')} (chunk {doc.metadata.get('position', 'N/A')})" for i, doc in enumerate(source_documents)]
    full_response = f"{answer}\n\n**Sources:**\n" + "\n".join(citations)
    
    return full_response, source_documents

def get_vector_count(collection_name="user-data-collection"):
    """
    Returns the number of vectors in the Qdrant collection.
    """
    client = qdrant_client.QdrantClient(qdrant_url, api_key=qdrant_api_key)
    try:
        count = client.count(collection_name=collection_name, exact=True).count
    except Exception:
        # If collection doesn't exist, count is 0
        count = 0
    return count

def delete_all_points_from_collection(collection_name="user-data-collection"):
    """
    Deletes all points from the Qdrant collection without deleting the collection.
    """
    client = qdrant_client.QdrantClient(qdrant_url, api_key=qdrant_api_key)
    try:
        # Deletes all points in a collection
        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_client.http.models.FilterSelector(
                filter=qdrant_client.http.models.Filter(
                    must=[],
                )
            )
        )
        print(f"All points deleted from collection '{collection_name}'.")
    except Exception as e:
        print(f"Could not delete points: {e}")

# --- Streamlit App ---

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot: Qdrant, Gemini, and Cohere")

# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    text_input = st.text_area("Or paste your text here", height=200)
    
    if st.button("Process and Store Data"):
        if uploaded_file is not None:
            text_data = uploaded_file.read().decode("utf-8")
            st.info("Processing uploaded file.")
        elif text_input.strip():
            text_data = text_input
            st.info("Processing pasted text.")
        else:
            st.warning("Please upload a file or paste text to process.")
            st.stop()

        with st.spinner("Chunking and upserting data to Qdrant..."):
            start_time = time.time()
            documents, metadatas = process_and_chunk_data(text_data)
            try:
                upsert_data_to_qdrant(documents, metadatas)
                end_time = time.time()
                st.success(f"Successfully processed and stored {len(documents)} chunks in {end_time - start_time:.2f} seconds.")
                st.session_state.data_processed = True
            except Exception as e:
                import traceback
                st.error(f"Error upserting data to Qdrant: {str(e)}")
                traceback.print_exc()
                st.stop()

    st.header("Database Management")
    vector_count = get_vector_count()
    st.info(f"Vectors in database: {vector_count}")

    if st.button("Clear Database"):
        with st.spinner("Deleting all points from collection..."):
            delete_all_points_from_collection()
            st.success("All points deleted from the collection.")
            st.rerun()

# --- Main Chat Interface ---
st.header("Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

if not st.session_state.data_processed:
    st.warning("Please process some data first using the sidebar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("What is your question?"):
    if not st.session_state.data_processed:
        st.warning("Please process data before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            # Retrieval
            retriever = get_retriever()
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Reranking
            reranked_docs = rerank_documents(query, retrieved_docs)
            
            # Answering
            chain = get_conversational_chain()
            # We need to pass the reranked docs as context to the chain
            # The current chain setup doesn't directly support this.
            # Let's modify how we generate the response.
            
            context = "\n\n".join([doc.page_content for doc in reranked_docs])
            prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
            """
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            from langchain.chains.llm import LLMChain
            llm_chain = LLMChain(llm=model, prompt=prompt)
            answer = llm_chain.run({"context": context, "question": query})

            # Create citations from reranked docs
            citations = [f"[{i+1}] {doc.metadata.get('title', 'Unknown Title')} (chunk {doc.metadata.get('position', 'N/A')})" for i, doc in enumerate(reranked_docs)]
            full_response = f"{answer}\n\n**Sources:**\n" + "\n".join(citations)

            end_time = time.time()

            st.markdown(full_response)
            
            # Display timing and cost (rough estimate)
            st.info(f"Response generated in {end_time - start_time:.2f} seconds.")
            # Rough token count and cost - replace with actuals if needed
            # This is a very rough estimation
            input_tokens = len(query.split()) + len(context.split())
            output_tokens = len(answer.split())
            st.info(f"Estimated tokens: {input_tokens + output_tokens}")


    st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.expander("Show Reranked Sources"):
        for i, doc in enumerate(reranked_docs):
            st.write(f"**Source [{i+1}]**")
            st.write(f"**Title:** {doc.metadata.get('title', 'N/A')}")
            st.write(f"**Chunk Position:** {doc.metadata.get('position', 'N/A')}")
            st.write(doc.page_content)
            st.divider()
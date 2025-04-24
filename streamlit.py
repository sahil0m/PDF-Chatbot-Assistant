import streamlit as st
import time
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions and classes from main.py
from main import (
    ModelManager,
    upload_pdfs,
    create_vector_store,
    retrieve_docs,
    question_pdf,
)

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styles
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-section {
        border: 2px dashed #4e8df5;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .main-header {
        color: #4e8df5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>PDF Chat Assistant</h1>", unsafe_allow_html=True)

# Session state initialization
if "db" not in st.session_state:
    st.session_state.db = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for file upload and processing
with st.sidebar:
    st.header("Document Upload")
    with st.container():
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        if st.button("Process Documents", key="process_docs"):
            with st.spinner("Processing documents..."):
                model_manager = ModelManager()
                start_time = time.time()
                documents = upload_pdfs(uploaded_files)
                db = create_vector_store(documents)
                st.session_state.documents = documents
                st.session_state.db = db
                st.session_state.file_processed = True
                processing_time = time.time() - start_time
                st.markdown(
                    f"<div class='status-box success-box'>✅ Processed {len(documents)} document segments in {processing_time:.2f} seconds</div>",
                    unsafe_allow_html=True,
                )

    # Model status display
    model_manager = ModelManager()
    if model_manager.embeddings and model_manager.llm:
        st.markdown(
            "<div class='status-box info-box'>✅ Models loaded and ready</div>",
            unsafe_allow_html=True,
        )
    elif model_manager.is_loading:
        st.markdown(
            "<div class='status-box info-box'>⏳ Loading models...</div>",
            unsafe_allow_html=True,
        )

# Main interface for questions
if st.session_state.file_processed:
    st.header("Ask Questions About Your Documents")
    query = st.text_input(
        "Enter your question:",
        key="query",
        placeholder="What does the document say about...?",
    )

    if query:
        with st.spinner("Searching documents and generating answer..."):
            docs = retrieve_docs(st.session_state.db, query)
            answer = question_pdf(query, docs)
            st.session_state.chat_history.append((query, answer))

    if st.session_state.chat_history:
        st.subheader("Conversation")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
            st.divider()

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
else:
    st.info("👈 Please upload PDF documents in the sidebar and click 'Process Documents' to get started.")

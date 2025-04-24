# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM


# embeddings = OllamaEmbeddings(model="llava-phi3")
# model = OllamaLLM(model="llava-phi3")

# template = """
# You are a ChatBot designed to interact with the extracted information and your job is to provide concise, accurate answers. Given the information below, answer the question precisely and with only the required output to be specific . If the answer is not available in the whole document, respond with 'I don't know.'

# Question: {question}
# Context: {context}
# Answer:
# """


# def upload_pdfs(files):
#     all_documents = []
#     for file in files:
#         with open(file.name, "wb") as f:
#             f.write(file.getbuffer())
#         loader = PyPDFLoader(file.name)
#         documents = loader.load()
#         all_documents.extend(documents)
#     return all_documents


# # Create a vector store from the documents
# def create_vector_store(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000, chunk_overlap=300, add_start_index=True
#     )
#     chunked_docs = text_splitter.split_documents(documents)
#     db = FAISS.from_documents(chunked_docs, embeddings)
#     return db


# # Retrieve relevant documents based on a query
# def retrieve_docs(db, query, k=4):
#     return db.similarity_search(query, k)


# def question_pdf(question, documents):
#     context = "\n\n".join([doc.page_content for doc in documents])
#     prompt = ChatPromptTemplate.from_template(template)
#     chain = prompt | model
#     response = chain.invoke({"question": question, "context": context})

#     # Strip the response to keep only the text after the 'think' tag
#     if "<think>" in response:
#         response = response.split("</think>", 1)[1].strip()

#     return response


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile
import time
import threading
import concurrent.futures
from functools import lru_cache


# Use singleton pattern for model loading to avoid reloading
class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance.embeddings = None
                cls._instance.llm = None
                cls._instance.is_loading = False
                cls._instance.load_event = threading.Event()
            return cls._instance

    def get_embeddings(self):
        if self.embeddings is None and not self.is_loading:
            self._load_models()
        self.load_event.wait()  # Wait for models to load if in progress
        return self.embeddings

    def get_llm(self):
        if self.llm is None and not self.is_loading:
            self._load_models()
        self.load_event.wait()  # Wait for models to load if in progress
        return self.llm

    def _load_models(self):
        self.is_loading = True
        # Load models in a separate thread
        threading.Thread(target=self._load_models_thread).start()

    def _load_models_thread(self):
        try:
            # Set lower temperature for more precise answers
            self.embeddings = OllamaEmbeddings(model="llama2:latest")
            self.llm = OllamaLLM(model="llama2:latest", temperature=0.1)
            self.is_loading = False
            self.load_event.set()
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loading = False


# Use optimized template with more specific instructions
TEMPLATE = """
You are a specialized document assistant. Your task is to answer questions based solely on the provided context.
Rules:
1. Only use information from the context
2. If the answer isn't in the context, say "I don't have enough information to answer this question."
3. Keep answers concise and direct
4. Do not hallucinate or add information not present in the context

Question: {question}
Context: {context}

Answer:
"""


# Optimized PDF upload function with concurrent processing
def upload_pdfs(files):
    all_documents = []
    temp_files = []

    # Create temporary files to avoid file handle issues
    for file in files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(file.getbuffer())
        temp_file.close()
        temp_files.append(temp_file.name)

    # Use thread pool for concurrent PDF loading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(load_pdf, file_path): file_path for file_path in temp_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                documents = future.result()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading PDF: {e}")

    # Clean up temporary files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

    return all_documents


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# Create vector store with optimized parameters
def create_vector_store(documents):
    start_time = time.time()

    # Optimize text splitting parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks for better retrieval
        chunk_overlap=150,  # Reduced overlap for faster processing
        add_start_index=True,
    )
    chunked_docs = text_splitter.split_documents(documents)

    # Get the embeddings instance
    model_manager = ModelManager()
    embeddings = model_manager.get_embeddings()

    # Create vector store
    db = FAISS.from_documents(chunked_docs, embeddings)

    print(f"Vector store created in {time.time() - start_time:.2f} seconds")
    return db


# Optimized document retrieval
def retrieve_docs(db, query, k=4):
    start_time = time.time()
    docs = db.similarity_search(query, k=k)
    print(f"Document retrieval completed in {time.time() - start_time:.2f} seconds")
    return docs


# Optimized question answering
def question_pdf(question, documents):
    start_time = time.time()

    # Get the LLM instance
    model_manager = ModelManager()
    model = model_manager.get_llm()

    # Prepare context with document scores and content
    formatted_docs = []
    for i, doc in enumerate(documents):
        formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")

    context = "\n\n".join(formatted_docs)

    # Create chain
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | model

    # Get response
    response = chain.invoke({"question": question, "context": context})

    # Clean up response
    if "<think>" in response:
        response = response.split("</think>", 1)[1].strip()

    print(f"Answer generated in {time.time() - start_time:.2f} seconds")
    return response

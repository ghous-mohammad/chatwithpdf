import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from langchain.embeddings.base import Embeddings
from typing import List

# Load environment variables
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API")
hf_api_key = os.environ.get("HF_API")

# Custom Embeddings class for Hugging Face Inference API
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_url: str):
        self.api_key = api_key
        self.model_url = model_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(self.model_url, headers=headers, json={"inputs": texts})
            response.raise_for_status()
            embeddings = response.json()
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Initialize Hugging Face embeddings
hf_embeddings = HuggingFaceEmbeddings(
    api_key=hf_api_key,
    model_url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    if not text.strip():
        st.error("No text extracted from PDFs. Please check the files.")
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        if not chunks:
            st.error("No text chunks created. Text may be empty or invalid.")
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return []

def get_vector_store(text_chunks):
    try:
        # Ensure faiss_index directory exists
        os.makedirs("faiss_index", exist_ok=True)
        vector_store = FAISS.from_texts(text_chunks, embedding=hf_embeddings)
        vector_store.save_local("faiss_index")
        st.write("FAISS index created and saved successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

def get_conversational_chain():
    prompt_template = """
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    try:
        model = ChatOpenAI(
            model_name="gemini-2.0-flash",
            openai_api_key=gemini_api_key,
            openai_api_base="https://generativelanguage.googleapis.com/v1beta",
            temperature=0.3
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def user_input(user_question):
    try:
        # Check if FAISS index exists
        if not os.path.exists("faiss_index/index.faiss"):
            st.error("FAISS index not found. Please upload PDFs and click 'Submit & Process' first.")
            return
        new_db = FAISS.load_local("faiss_index", hf_embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        if chain is None:
            return
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Initialize session state to track processing status
    if "processed" not in st.session_state:
        st.session_state.processed = False

    user_question = st.text_input("Ask a Question from the PDF Files"  )

    if user_question:
        if not st.session_state.processed:
            st.error("Please upload PDFs and click 'Submit & Process' before asking questions.")
        else:
            user_input(user_question)
            

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Clear previous FAISS index to avoid conflicts
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                if not pdf_docs:
                    st.error("No PDFs uploaded. Please select at least one PDF.")
                    return
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    return
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    return
                get_vector_store(text_chunks)
                st.session_state.processed = True
                st.success("Done")

if __name__ == "__main__":
    main()
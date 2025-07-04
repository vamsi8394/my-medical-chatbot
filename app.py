import streamlit as st
import os
import glob
import json
import logging
import time
from datetime import datetime
import re
import pdfplumber
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set in the environment variables.")
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
BOOKS_FOLDER = "LLM_DB/"
DEFAULT_CHUNK_SIZE = 2028
DEFAULT_CHUNK_OVERLAP = 256

DEFAULT_PROMPT_TEMPLATE = """
As your Virtual Health Assistant... (shortened for brevity)
{context}
{question}
"""

@st.cache_data
def extract_clean_text(pdf_folder):
    text = ""
    for pdf_path in glob.glob(os.path.join(pdf_folder, '*.pdf')):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = re.sub(r"Page \d+", "", page_text)
                        page_text = re.sub(r"\n{2,}", "\n", page_text)
                        text += page_text
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {str(e)}")
    return text

@st.cache_data
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    return splitter.split_text(text)

@st.cache_data
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    vector_store.save_local("faiss_index")
    return vector_store

@st.cache_data
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def get_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro-001", temperature=0.2)
    prompt = PromptTemplate(template=DEFAULT_PROMPT_TEMPLATE, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def answer_question(question):
    start_time = time.time()
    db = load_vector_store()
    docs = db.similarity_search_by_score(question, k=5)
    chain = get_chain()
    result = chain({"input_documents": [doc for doc, _ in docs], "question": question})
    duration = time.time() - start_time
    st.write(f"\n**Response Time:** {duration:.2f} seconds")
    return result["output_text"], docs


def main():
    st.set_page_config(page_title="Krishna Health Chatbot", layout="wide")
    st.title("📘 Krishna Health AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    menu = st.sidebar.radio("Navigate", ["Chatbot", "Upload PDFs", "Dashboard"])

    if menu == "Upload PDFs":
        files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type="pdf")
        if files:
            for file in files:
                with open(os.path.join(BOOKS_FOLDER, file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded.")
            if st.button("Process Text"):
                text = extract_clean_text(BOOKS_FOLDER)
                chunks = split_text(text)
                create_vector_store(chunks)
                st.success("Text processed and embeddings created.")

    elif menu == "Chatbot":
        question = st.text_input("Ask a medical question:")
        if question:
            response, docs = answer_question(question)
            st.write("\n**AI Response:**")
            st.write(response)
            with st.expander("Sources"):
                for doc, score in docs:
                    st.markdown(f"**Score:** {score:.4f}\n\n{doc.page_content[:300]}...")
            st.session_state.chat_history.append((question, response))

    elif menu == "Dashboard":
        st.subheader("Session History")
        for i, (q, r) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}\n\n**A{i+1}:** {r}")

    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px;">
            <strong>Developed by Krishna</strong><br>
            <img src="https://raw.githubusercontent.com/vamsi8394/my-medical-chatbot/main/assets/chatbot-screenshot.png" alt="App Screenshot" width="500">
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

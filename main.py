import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generate_gemini_response(question, context):
    """Generate response using Google Gemini API"""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Context: {context}\n\nQuestion: {question}")
    return response.text

# Streamlit UI Enhancement
st.set_page_config(page_title="Ask your PDF with Gemini", layout="wide")
st.markdown(
    """
    <style>
        body {background-color: #121212; color: white;}
        .main {background-color: #1e1e1e; padding: 2rem; border-radius: 10px; color: white;}
        .stButton>button {border-radius: 10px; background-color: #007bff; color: white;}
        .stTextInput>div>div>input {border-radius: 10px; padding: 10px; background-color: #333; color: white;}
    </style>
    """, unsafe_allow_html=True
)

st.title("📄 Ask your PDF (Powered by Gemini)")
st.sidebar.header("How to Use")
st.sidebar.write("Upload a PDF file, ask a question, and get AI-powered answers!")

# Upload PDF
pdf = st.file_uploader("📂 Upload your PDF", type="pdf")

if pdf is not None:
    with st.spinner("Processing PDF..."):
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        
        # Display PDF details
        num_pages = len(pdf_reader.pages)
        st.sidebar.success(f"✅ File uploaded: {pdf.name} ({num_pages} pages)")

        # Split text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="local_model")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User Question
        user_question = st.text_input("💡 Ask a question about your PDF:", placeholder="Type your question here...")
        
        if user_question:
            with st.spinner("Searching for answers..."):
                docs = knowledge_base.similarity_search(user_question)
                context = " ".join([doc.page_content for doc in docs])
                
                # Get AI response
                response = generate_gemini_response(user_question, context)
                
                # Display response in a chat-like format
                st.markdown("### 🤖 AI Response")
                st.info(response)

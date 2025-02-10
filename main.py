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

def main():
    st.set_page_config(page_title="Ask your PDF with Gemini")
    st.header("Ask your PDF 💬 (Powered by Gemini)")

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings (Using HuggingFace instead of OpenAI)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User Question
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            context = " ".join([doc.page_content for doc in docs])

            # Get answer from Gemini
            response = generate_gemini_response(user_question, context)
            st.write(response)

if __name__ == '__main__':
    main()

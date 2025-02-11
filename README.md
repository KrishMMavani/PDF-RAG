# 📄 Ask Your PDF (Powered by Gemini)

## Overview
This project allows users to upload a PDF file and ask questions about its content. The system processes the PDF, extracts text, generates embeddings, and provides AI-powered responses using Google's Gemini API.

## Features
- 📂 Upload PDFs and extract text automatically
- 💡 Ask questions about the PDF content
- 🤖 AI-powered answers using Gemini API
- 🌙 Dark mode UI for a better experience
- 🚀 Fast and efficient similarity search with FAISS

## Models Used
- `HuggingFaceEmbeddings`: `sentence-transformers/all-MiniLM-L6-v2`
- `FAISS` (Facebook AI Similarity Search) for vector-based retrieval
- `GenerativeModel`: `gemini-pro` (Google Gemini API)

## APIs Used
- `Google Generative AI API` (Gemini)

## Process Diagram
![Flow Chart](https://github.com/user-attachments/assets/f3075a3a-915e-432f-8e4e-29ccd632ddfe)


## Installation & Setup
### Step 1: Install Dependencies
```sh
pip install -r requirements.txt
```

### Step 2: Setup env file
```sh
GEMINI_API_KEY="your GEMINI API"
```

### Step 3: Download the model for embedding
```sh
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("local_model")
```

### Step 4: Run the Project
```sh
streamlit run main.py
```

### Preview

![Screenshot 2025-02-11 101002](https://github.com/user-attachments/assets/5d7809bb-f918-4d95-9f9b-e9bc9f374561)




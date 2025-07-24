# 🤖 ClaimIQ: AI-Powered Insurance Claim Analysis using RAG

Welcome to **ClaimIQ** — a Retrieval-Augmented Generation (RAG) chatbot that analyzes insurance claim documents and queries using advanced NLP techniques. Upload claim PDFs, ask questions, or retrieve relevant past cases, all in a clean and interactive interface built with Streamlit.

---

## 📌 Features

✅ Extracts text from insurance claim PDFs  
✅ Retrieves similar past claims using semantic search (FAISS + Sentence Transformers)  
✅ Summarizes and analyzes claims with an LLM (BART)  
✅ Lets users ask free-form insurance queries  
✅ Lightweight and fast — runs on CPU  

---

## 🧠 How It Works

### 🧾 Input
- User uploads a new claim PDF or types a custom query.

### 🔍 Retrieval
- Embeds user input using `all-MiniLM-L6-v2` (SentenceTransformers).
- Uses FAISS to retrieve top-k similar past claims.

### 🧪 Generation
- Passes context + query to `facebook/bart-large-cnn` summarization model.
- Returns a natural language summary or response.

---

## 🛠️ Tech Stack

| Layer            | Tool / Library                       |
|------------------|--------------------------------------|
| UI               | [Streamlit](https://streamlit.io)    |
| Text Extraction  | PyMuPDF (`fitz`)                     |
| Embedding Model  | `all-MiniLM-L6-v2` (HuggingFace)     |
| Vector Search    | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM Generator    | `facebook/bart-large-cnn`            |
| Language         | Python 🐍                             |

---

## 🚀 Demo

https://user-demo-link.com (optional)

<p align="center">
  <img src="assets/demo.gif" alt="demo" width="80%">
</p>

---

## 📂 File Structure


import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# --------- TEXT EXTRACTION FROM PDF ---------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --------- LOAD SAMPLE CLAIM DATABASE ---------
def load_sample_claims():
    return [
        "The policyholder reported a car accident involving rear-end collision. Damage to the bumper and trunk area. Estimated cost: $2,500.",
        "Claim submitted for roof damage due to heavy hailstorm. Inspector verified shingles missing. Repair estimate: $7,200.",
        "Fire damage in the kitchen due to stove malfunction. Smoke damage reported in adjacent rooms. Loss estimated at $14,000.",
        "Flooding in basement after pipe burst. Water damage to flooring and furniture. Estimated repair: $4,800.",
        "Theft claim for stolen jewelry and electronics from home. Police report filed. Loss estimated at $9,500.",
    ]

# --------- BUILD VECTOR SEARCH INDEX ---------
@st.cache_resource
def build_vector_store(claims):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(claims, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    return index, embedder, claims

# --------- FIND SIMILAR CLAIMS ---------
def retrieve_similar(text, index, embedder, documents, top_k=3):
    query_vec = embedder.encode([text], convert_to_numpy=True).astype('float32')
    D, I = index.search(query_vec, top_k)
    return [documents[i] for i in I[0]]

# --------- LOAD SUMMARIZER MODEL ---------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# --------- GENERATE ANALYSIS OR RESPONSE ---------
def generate_analysis(current_text, retrieved_claims, summarizer):
    context = "\n\n".join(retrieved_claims)
    input_text = f"Context:\n{context}\n\nNew Claim or Query:\n{current_text}\n\nAnalyze or Answer:"
    result = summarizer(input_text, max_length=300, min_length=100, do_sample=False)
    return result[0]['summary_text']

# --------- STREAMLIT UI ---------
st.set_page_config(page_title="ClaimIQ - AI Insurance Assistant", layout="wide")
st.title("🤖 ClaimIQ: Insurance Claim Analysis with RAG")

# --------- CLAIM PDF UPLOAD ---------
uploaded_file = st.file_uploader("📄 Upload a new insurance claim PDF", type="pdf")

if uploaded_file:
    with st.spinner("📑 Extracting text from PDF..."):
        new_claim_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📃 Extracted Claim Text")
    st.text_area("Claim Content", new_claim_text[:2000], height=200)

    if st.button("🚀 Analyze Claim"):
        with st.spinner("🔍 Analyzing..."):
            past_claims = load_sample_claims()
            vector_index, embedder, documents = build_vector_store(past_claims)
            similar_claims = retrieve_similar(new_claim_text, vector_index, embedder, documents)
            summarizer = load_summarizer()
            analysis = generate_analysis(new_claim_text, similar_claims, summarizer)

        st.subheader("🧠 AI-Powered Claim Analysis")
        st.success(analysis)

        st.subheader("📚 Similar Past Claims Used")
        for i, claim in enumerate(similar_claims, 1):
            st.markdown(f"*Claim #{i}:* {claim}")

# --------- FREE QUERY SECTION ---------
st.markdown("---")
st.subheader("💬 Or Ask Your Own Insurance Query")

user_query = st.text_input("🔎 Enter your query below")

if user_query:
    if st.button("🚀 Analyze Query"):
        with st.spinner("🔍 Retrieving and Analyzing..."):
            past_claims = load_sample_claims()
            vector_index, embedder, documents = build_vector_store(past_claims)
            similar_claims = retrieve_similar(user_query, vector_index, embedder, documents)
            summarizer = load_summarizer()
            query_response = generate_analysis(user_query, similar_claims, summarizer)

        st.subheader("📘 Answer to Your Query")
        st.success(query_response)

        st.subheader("📚 Relevant Past Claims Used")
        for i, claim in enumerate(similar_claims, 1):
            st.markdown(f"*Context #{i}:* {claim}")
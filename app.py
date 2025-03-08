import os
import faiss
import numpy as np
import pickle
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from streamlit_extras.add_vertical_space import add_vertical_space
import re
# ✅ Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyBciC9FK5U2e3pni6h6Yq2zZolLKVUZBu0"  # 🔹 Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Define Paths
FAISS_INDEX_PATH = r"E:\PythonProject\BiologyBOT\faiss_index\biology_textbook.faiss"
CHUNKS_PATH = r"E:\PythonProject\BiologyBOT\faiss_index\biology_textbook_chunks.pkl"
METADATA_PATH = r"E:\PythonProject\BiologyBOT\faiss_index\biology_textbook_metadata.pkl"

# ✅ Load FAISS Index
index = faiss.read_index(FAISS_INDEX_PATH)

# ✅ Load Text Chunks & Metadata
with open(CHUNKS_PATH, "rb") as f:
    text_chunks = pickle.load(f)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)


# ✅ Function to Classify Question Type (Factual vs. Conceptual)
def classify_question_type(query: str, faiss_index, text_chunks, k=3):
    """Determines if a question is factual or conceptual based on FAISS search results."""
    query_embedding = embedding_model.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding), k)

    retrieved_texts = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]

    return "factual" if retrieved_texts else "conceptual"

def retrieve_relevant_text(query, k=5):
    """Retrieve the most relevant textbook chunks from FAISS."""
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)

    retrieved_texts = []
    retrieved_pages = []

    for idx in indices[0]:
        if idx < len(text_chunks):
            retrieved_texts.append(text_chunks[idx])  # ✅ Store only the clean text

            # ✅ Extract only numeric page number
            raw_page = metadata[idx]["page"]
            clean_page = re.sub(r"[^\d]", "", str(raw_page))  # ✅ Remove non-numeric characters

            if clean_page.isdigit():  # ✅ Ensure it’s a valid number
                retrieved_pages.append(int(clean_page))

    combined_text = "\n\n".join(retrieved_texts)
    return combined_text, retrieved_pages  # ✅ Now returns properly formatted pages


# ✅ Function to Generate Answer Using Google Gemini API
def get_answer_from_gemini(query, context, question_type):
    """Generates an answer using Google Gemini API, using textbook content for factual questions."""

    if question_type == "factual":
        system_prompt = f"""
        You are a highly knowledgeable biology tutor. Use the provided textbook content to answer the question accurately.

        If the exact answer is not explicitly stated, infer the best possible response using the information available.

        Always provide structured, easy-to-understand explanations suitable for a 9th-grade student.

        If required, include a **real-world example** to improve understanding.

        ### TEXTBOOK CONTENT:
        {context}

        ### QUESTION: {query}

        ### DETAILED ANSWER:
        """
    else:
        system_prompt = f"""
        You are an expert biology tutor for 9th-grade students.

        Provide a clear, well-structured explanation for the question below, using simple and easy-to-understand language. 
        If possible, **include a real-world example or analogy** to make the concept more relatable.

        Ensure your response is scientifically accurate and engaging.

        ### QUESTION: {query}

        ### EXPLANATION:
        """

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(system_prompt)
        return response.text if response.text else "Error: Empty response from Gemini."
    except Exception as e:
        return f"Error calling Gemini API: {e}"


# ✅ Improved Streamlit UI
st.set_page_config(page_title="Biology Chatbot", layout="wide", page_icon="📖")

# 🔹 Sidebar with Instructions
with st.sidebar:
    st.title("📖 Biology Chatbot")
    st.markdown("""
    **How to Use:**
    - Ask me anything from your 9th Class Biology textbook.
    - If an answer exists in the textbook, I will retrieve it.
    - If not, I'll explain the concept in simple terms.
    """)
    add_vertical_space(2)
    st.caption("🔹 Powered by FAISS + Gemini AI")

# 🔹 Main Chatbot UI
st.title("💡 Biology Chatbot (AI Tutor)")
st.markdown("Ask your questions below! I'll retrieve textbook content or explain concepts clearly.")

# ✅ Customizing UI with Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Display Chat History
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**👨‍🎓 You:** {chat['user']}")

    with st.chat_message("assistant"):
        st.markdown(f"**🤖 Bot:** {chat['bot']}")

        if chat["pages"]:  # ✅ Ensure proper display of "Page X"
            st.markdown("📖 **Source Pages:**")
            unique_pages = sorted(set(chat["pages"]))  # ✅ Remove duplicates, sort pages
            for page in unique_pages:
                st.markdown(f"📄 **Page {page}**")  # ✅ Now properly formatted

# ✅ Floating Chat Input
user_query = st.chat_input("Type your question here...")

if user_query:
    with st.spinner("🔍 Searching textbook..."):
        question_type = classify_question_type(user_query, index, text_chunks)
        retrieved_text, retrieved_pages = retrieve_relevant_text(user_query)

        answer = get_answer_from_gemini(user_query, retrieved_text, question_type)

        # ✅ Store conversation history
        st.session_state.chat_history.append({"user": user_query, "bot": answer, "pages": retrieved_pages})

        # ✅ Refresh UI with New Message
        st.rerun()

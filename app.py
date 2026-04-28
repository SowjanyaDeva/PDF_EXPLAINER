# ============================================================
# 📄 Chat with PDF — Main Streamlit App
# ============================================================
# This is the entry point. Streamlit runs this file top-to-bottom
# every time the user interacts with the UI.
# ============================================================

import streamlit as st
from pdf_processor import process_pdf
from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with Your PDF")
st.caption("Upload a PDF and ask questions about it using AI")

# ── Session State ─────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction.
# st.session_state lets us persist data across those re-runs.
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None       # Our RAG engine (built after upload)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # List of {"role": ..., "content": ...}

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False   # Have we processed a PDF yet?

# ── Sidebar: Upload & Settings ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Max size: ~50MB. Larger PDFs take longer to process."
    )

    # Model selector — these are Ollama model names
    model_name = st.selectbox(
        "Choose Ollama Model",
        options=["llama3.2", "mistral", "gemma2", "llama3.1"],
        help="Make sure this model is pulled in Ollama first (e.g. `ollama pull llama3.2`)"
    )

    # Chunk size: how many characters per text chunk
    # Smaller = more precise retrieval | Larger = more context per chunk
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200, max_value=2000, value=800, step=100,
        help="How big each text chunk is. Smaller = more precise, Larger = more context."
    )

    # Top-K: how many chunks to retrieve per question
    top_k = st.slider(
        "Chunks to Retrieve (Top-K)",
        min_value=1, max_value=10, value=4,
        help="How many text chunks to send to the AI as context."
    )

    st.divider()

    # Process button — only active when a file is uploaded
    if uploaded_file:
        if st.button("🔄 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Reading and chunking your PDF..."):
                # Step 1: Extract text from the PDF and split into chunks
                chunks = process_pdf(uploaded_file, chunk_size=chunk_size)

            with st.spinner(f"Building vector index with {len(chunks)} chunks..."):
                # Step 2: Create embeddings for every chunk and store in ChromaDB
                engine = RAGEngine(model_name=model_name)
                engine.index_chunks(chunks)
                st.session_state.rag_engine = engine
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []  # Clear old chat on new PDF

            st.success(f"✅ Done! Indexed {len(chunks)} chunks.")

    # Info box
    st.info(
        "**How it works:**\n\n"
        "1. Your PDF text is split into small chunks\n"
        "2. Each chunk is turned into a vector (embedding)\n"
        "3. When you ask a question, the most relevant chunks are retrieved\n"
        "4. Those chunks + your question are sent to the LLM\n\n"
        "This is called **RAG** (Retrieval-Augmented Generation)."
    )

# ── Main Area: Chat Interface ─────────────────────────────────
if not st.session_state.pdf_processed:
    # Show a placeholder before any PDF is uploaded
    st.info("👈 Upload a PDF in the sidebar to get started.")

    # Demo explanation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Choose any PDF — a research paper, a contract, a textbook chapter.")
    with col2:
        st.markdown("### 2️⃣ Process")
        st.markdown("Click **Process PDF**. The app chunks the text and builds a search index.")
    with col3:
        st.markdown("### 3️⃣ Ask")
        st.markdown("Type any question. The AI finds the relevant parts and answers from them.")

else:
    # ── Chat History Display ──────────────────────────────────
    # Loop through all previous messages and display them
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat Input ────────────────────────────────────────────
    # st.chat_input stays pinned at the bottom of the page
    if user_question := st.chat_input("Ask a question about your PDF..."):

        # 1. Show the user's message in the chat
        with st.chat_message("user"):
            st.markdown(user_question)

        # 2. Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # 3. Generate the AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching PDF and generating answer..."):
                answer, source_chunks = st.session_state.rag_engine.query(
                    user_question,
                    top_k=top_k
                )

            # Display the answer
            st.markdown(answer)

            # Optionally show what chunks were used (great for learning!)
            with st.expander("🔍 Source chunks used (click to inspect)"):
                for i, chunk in enumerate(source_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk)
                    st.divider()

        # 4. Save assistant reply to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
# ============================================================
# 🧠 rag_engine.py — The Core RAG Logic
# ============================================================
#
# SUPPORTS TWO MODES — controlled by the USE_GROQ flag below:
#
#   LOCAL (USE_GROQ = False):
#     Uses Ollama — great for development on your own PC.
#     Requires Ollama to be running: `ollama serve`
#
#   CLOUD (USE_GROQ = True):
#     Uses Groq API — free, fast, hostable on Hugging Face Spaces.
#     Requires a free API key from https://console.groq.com
#     Set it as an environment variable: GROQ_API_KEY=your_key
#
# Everything else (ChromaDB, embeddings, chunking) is identical.
# Only the final LLM call changes between the two modes.
# ============================================================

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Tuple

# ── Toggle this to switch between local and cloud mode ───────
# False = use Ollama locally
# True  = use Groq API (for Hugging Face / cloud deployment)
USE_GROQ = os.environ.get("USE_GROQ", "false").lower() == "true"

# Load the right LLM client based on the mode
if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
else:
    import ollama

from typing import List, Tuple


class RAGEngine:
    """
    Handles all the AI logic:
    - Embedding chunks and storing them in ChromaDB
    - Retrieving the most relevant chunks for a question
    - Calling Ollama to generate an answer
    """

    def __init__(self, model_name: str = "llama3.2"):
        """
        Set up ChromaDB and the embedding model.

        ChromaDB is a local vector database — it stores your text chunks
        as mathematical vectors (embeddings) and can find the most similar
        ones very quickly.

        We use "all-MiniLM-L6-v2" for embeddings because:
        - It's small and fast (runs on CPU, no GPU needed)
        - It's great at semantic similarity tasks
        - It's free and works offline via sentence-transformers
        """
        self.model_name = model_name

        # ChromaDB in-memory client — data lives in RAM (perfect for demos)
        # For production you'd use: chromadb.PersistentClient(path="./chroma_db")
        self.chroma_client = chromadb.Client()

        # The embedding function converts text → vector (list of floats)
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # A "collection" is like a table in a database — stores chunks + their vectors
        # We delete and recreate it so re-uploading a PDF starts fresh
        try:
            self.chroma_client.delete_collection("pdf_chunks")
        except Exception:
            pass  # Collection didn't exist yet — that's fine

        self.collection = self.chroma_client.create_collection(
            name="pdf_chunks",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
        )

    def index_chunks(self, chunks: List[str]) -> None:
        """
        Embed all text chunks and store them in ChromaDB.

        WHAT EMBEDDING MEANS:
            Text like "The cat sat on the mat" gets turned into a list of
            384 numbers like [0.023, -0.15, 0.87, ...]. These numbers capture
            the *meaning* of the text. Similar sentences → similar vectors.

        ChromaDB automatically calls our embed_fn on each chunk and stores
        both the raw text AND the vector so we can retrieve either later.
        """
        # Each document needs a unique ID — we just use "chunk_0", "chunk_1", etc.
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # Add all chunks in one batch call (much faster than one-by-one)
        self.collection.add(
            documents=chunks,  # The raw text
            ids=ids             # Unique identifiers
        )

    def retrieve(self, question: str, top_k: int = 4) -> List[str]:
        """
        Find the top-K most relevant chunks for the user's question.

        HOW SIMILARITY SEARCH WORKS:
            1. Embed the question into a vector
            2. Compute cosine similarity between the question vector
               and every chunk vector in ChromaDB
            3. Return the chunks with the highest similarity scores

        Cosine similarity measures the "angle" between two vectors.
        Vectors pointing in the same direction = similar meaning.
        """
        results = self.collection.query(
            query_texts=[question],  # ChromaDB embeds this automatically
            n_results=top_k          # How many chunks to return
        )

        # results["documents"] is a list of lists (one list per query)
        # We only have one query, so take [0]
        return results["documents"][0]

    def build_prompt(self, question: str, context_chunks: List[str]) -> str:
        """
        Build the prompt that we'll send to the LLM.

        PROMPT ENGINEERING PRINCIPLE:
            Give the model clear instructions + the relevant context.
            Tell it to stay grounded in the provided text so it doesn't
            hallucinate (make up things not in the PDF).
        """
        # Join all retrieved chunks into one context block
        context = "\n\n---\n\n".join(context_chunks)

        prompt = f"""You are a helpful assistant. Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I couldn't find that information in the PDF."

CONTEXT FROM THE PDF:
{context}

QUESTION:
{question}

ANSWER:"""

        return prompt

    def query(self, question: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """
        Full RAG pipeline for one user question.

        Returns:
            answer:        The LLM's text response
            source_chunks: The chunks that were retrieved (for transparency)
        """
        # Step 1: Find the most relevant chunks
        relevant_chunks = self.retrieve(question, top_k=top_k)

        # Step 2: Build a prompt with those chunks as context
        prompt = self.build_prompt(question, relevant_chunks)

        # Step 3: Call the LLM — Ollama locally, Groq on the cloud
        # This if/else is the ONLY thing that changes between the two modes
        if USE_GROQ:
            # Cloud mode — Groq API (free tier, very fast)
            # Get your free key at: https://console.groq.com
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",   # same Llama family as Ollama
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
        else:
            # Local mode — Ollama must be running (`ollama serve`)
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"]

        return answer, relevant_chunks
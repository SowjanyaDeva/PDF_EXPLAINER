# ============================================================
# 📚 pdf_processor.py — Step 1 of RAG: Extract & Chunk Text
# ============================================================
#
# WHY THIS STEP EXISTS:
#   LLMs have a limited "context window" — they can only read
#   a few thousand words at a time. A PDF might have 50+ pages.
#   Solution: split the PDF into small overlapping chunks,
#   then only feed the RELEVANT chunks to the LLM.
#
# WHAT THIS FILE DOES:
#   1. Reads a PDF file and extracts all the text
#   2. Splits it into overlapping chunks
#   3. Returns a list of chunk strings
# ============================================================

import fitz  # PyMuPDF — for reading PDF files
from typing import List


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Read a Streamlit UploadedFile and extract all text from it.

    PyMuPDF (fitz) opens the file in memory — no need to save it to disk.
    It reads each page and collects the text from every page.
    """
    # fitz.open() can take bytes directly with a filetype hint
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # page.get_text() returns the raw text of one page
        text = page.get_text()
        if text.strip():  # Skip blank pages
            all_text.append(text)

    doc.close()
    return "\n\n".join(all_text)  # Join pages with double newline


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split a long string into overlapping chunks.

    WHY OVERLAP?
        Imagine the answer to a question spans the end of chunk 3
        and the start of chunk 4. Without overlap, you'd only retrieve
        half the answer. Overlap ensures adjacent chunks share context.

    Args:
        text:       The full extracted text from the PDF
        chunk_size: How many characters per chunk (800 is a good default)
        overlap:    How many characters each chunk shares with the previous one

    Returns:
        A list of text strings (the chunks)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Don't cut a chunk mid-word — find the last space before `end`
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:  # Skip empty chunks
            chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        # This is what creates the overlap between consecutive chunks
        start += chunk_size - overlap

    return chunks


def process_pdf(uploaded_file, chunk_size: int = 800) -> List[str]:
    """
    Full pipeline: PDF file → list of text chunks.
    This is the function called from app.py.
    """
    # Step 1: Extract all text from every page
    full_text = extract_text_from_pdf(uploaded_file)

    # Step 2: Split into chunks (overlap = ~18% of chunk_size is a good rule of thumb)
    overlap = max(50, chunk_size // 5)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    return chunks
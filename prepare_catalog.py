"""
AUIS Academic Catalog Indexing Script

This script reads the AUIS Academic Catalog PDF, splits it into chunks,
generates embeddings, and stores them in a persistent ChromaDB collection.

Usage:
    python prepare_catalog.py
"""

import os

# Disable TensorFlow to avoid Keras compatibility issues
# We only need PyTorch for embeddings
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import re
from typing import List, Dict
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CATALOG_PDF_PATH,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL
)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing excessive whitespace
    and normalizing newlines.
    """
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_section_title(text: str) -> str:
    """
    Try to extract a section title from text.
    Simple heuristic: look for lines in ALL CAPS at the start.
    """
    lines = text.split('\n')
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if len(line) > 5 and line.isupper() and len(line) < 100:
            return line
    return "Catalog"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence or paragraph
        if end < text_len:
            # Look for paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start:
                end = paragraph_break
            else:
                # Look for sentence break
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('.\n', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_break > start:
                    end = sentence_break + 1
        
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Only keep chunks with substantial content
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < text_len else text_len
    
    return chunks


def read_and_chunk_pdf(pdf_path: str) -> List[Dict]:
    """
    Read PDF and return a list of chunk dictionaries with metadata.
    
    Returns:
        List of dicts with keys: text, page, chunk_id, section_title
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found at: {pdf_path}\n"
            f"Please place the AUIS Academic Catalog PDF at this location."
        )
    
    print(f"Reading PDF from: {pdf_path}")
    all_chunks = []
    chunk_counter = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text from page
            text = page.extract_text()
            
            if not text or len(text.strip()) < 50:
                # Skip pages with minimal text
                continue
            
            # Clean the text
            text = clean_text(text)
            
            # Attempt to extract section title
            section_title = extract_section_title(text)
            
            # Split into chunks
            text_chunks = chunk_text(text)
            
            # Create metadata for each chunk
            for chunk in text_chunks:
                chunk_counter += 1
                chunk_data = {
                    'text': chunk,
                    'page': page_num,
                    'chunk_id': chunk_counter,
                    'section_title': section_title
                }
                all_chunks.append(chunk_data)
            
            # Progress indicator
            if page_num % 10 == 0:
                print(f"Processed {page_num}/{total_pages} pages...")
    
    print(f"Extracted {len(all_chunks)} chunks from {total_pages} pages")
    return all_chunks


def create_chroma_index(chunks: List[Dict]):
    """
    Create embeddings and store chunks in ChromaDB.
    """
    print(f"\nInitializing embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
    # Create persistent Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "AUIS Academic Catalog chunks with embeddings"}
    )
    
    print(f"\nGenerating embeddings and storing {len(chunks)} chunks...")
    
    # Process in batches for efficiency
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Prepare data for this batch
        texts = [chunk['text'] for chunk in batch]
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in batch]
        metadatas = [
            {
                'page': chunk['page'],
                'chunk_id': chunk['chunk_id'],
                'section_title': chunk['section_title']
            }
            for chunk in batch
        ]
        
        # Generate embeddings
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        # Progress indicator
        print(f"Stored {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
    
    print(f"\n✓ Indexing complete!")
    print(f"✓ Stored {len(chunks)} chunks in {CHROMA_DB_PATH}/{COLLECTION_NAME}")


def main():
    """
    Main function to orchestrate the indexing process.
    """
    try:
        print("=" * 60)
        print("AUIS Academic Catalog Indexing")
        print("=" * 60)
        
        # Step 1: Read and chunk the PDF
        chunks = read_and_chunk_pdf(CATALOG_PDF_PATH)
        
        # Step 2: Create embeddings and store in ChromaDB
        create_chroma_index(chunks)
        
        print("\n" + "=" * 60)
        print("SUCCESS! The catalog index is ready.")
        print("You can now run: python app.py")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the AUIS Academic Catalog PDF is placed at:")
        print(f"    {os.path.abspath(CATALOG_PDF_PATH)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

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
import torch
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

# Batch size for processing chunks (larger = faster GPU utilization)
BATCH_SIZE = 150  # Balanced for 8GB VRAM


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text - optimized for speed.
    """
    # Replace multiple spaces but keep newlines for chunking
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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
    Optimized for speed with simpler boundary detection.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Only look for sentence breaks if not at the end
        if end < text_len:
            # Quick sentence break search (just period and space)
            sentence_break = text.rfind('. ', start, end)
            if sentence_break > start:
                end = sentence_break + 2
        
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Only keep chunks with substantial content
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < text_len else text_len
    
    return chunks


def add_batch_to_chroma(collection, embedding_model, batch: List[Dict], device: str):
    """
    Generate embeddings for a batch of chunks and add them to ChromaDB.
    This function processes and immediately stores batches to avoid memory buildup.
    
    Args:
        collection: ChromaDB collection object
        embedding_model: SentenceTransformer model
        batch: List of chunk dictionaries with text, page, chunk_id, section_title
        device: Device to use for embeddings ('cuda' or 'cpu')
    """
    if not batch:
        return
    
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
    
    # Generate embeddings on GPU if available
    # Use optimized batch_size for 8GB GPU
    embeddings = embedding_model.encode(
        texts, 
        convert_to_numpy=True, 
        device=device,
        show_progress_bar=False,
        batch_size=64,  # Optimized for 8GB VRAM
        normalize_embeddings=True  # Better for similarity search
    )
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    
    # Explicitly clear memory
    del embeddings, texts, ids, metadatas


def process_pdf_streaming(pdf_path: str, collection, embedding_model, device: str):
    """
    Stream-process the PDF: read page by page, chunk, embed, and store in batches.
    This avoids loading all chunks into memory at once.
    
    Args:
        pdf_path: Path to the PDF file
        collection: ChromaDB collection object
        embedding_model: SentenceTransformer model
        device: Device to use for embeddings ('cuda' or 'cpu')
        
    Returns:
        Total number of chunks processed
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found at: {pdf_path}\n"
            f"Please place the AUIS Academic Catalog PDF at this location."
        )
    
    print(f"Reading PDF from: {pdf_path}")
    
    chunk_counter = 0
    batch = []  # Current batch of chunks waiting to be embedded
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")
        print(f"Processing in batches of {BATCH_SIZE} chunks...\n")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                # Extract text from page
                text = page.extract_text()
                
                if not text or len(text.strip()) < 50:
                    # Skip pages with minimal text
                    continue
                
                # Clean the text
                text = clean_text(text)
                
                # Use simple section title (skip extraction for speed)
                section_title = f"Page {page_num}"
                
                # Split into chunks
                text_chunks = chunk_text(text)
                
                # Add each chunk to the current batch
                for chunk_text_content in text_chunks:
                    chunk_counter += 1
                    chunk_data = {
                        'text': chunk_text_content,
                        'page': page_num,
                        'chunk_id': chunk_counter,
                        'section_title': section_title
                    }
                    batch.append(chunk_data)
                    
                    # When batch is full, process it immediately
                    if len(batch) >= BATCH_SIZE:
                        add_batch_to_chroma(collection, embedding_model, batch, device)
                        batch.clear()  # Clear the batch after processing
                
                # Progress indicator every 50 pages (minimal printing overhead)
                if page_num % 50 == 0:
                    print(f"Processed {page_num}/{total_pages} pages... ({chunk_counter} chunks so far)")
                    
            except Exception as e:
                # Suppress warnings for speed (just continue)
                continue
        
        # Process any remaining chunks in the last batch
        if batch:
            add_batch_to_chroma(collection, embedding_model, batch, device)
            batch.clear()
        
        # Final progress
        print(f"Processed all {total_pages} pages... ({chunk_counter} total chunks)")
    
    return chunk_counter


def create_chroma_index_streaming(pdf_path: str):
    """
    Create embeddings and store chunks in ChromaDB using streaming/batched processing.
    """
    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    # Set to evaluation mode for faster inference
    embedding_model.eval()
    
    # Optimize for inference speed
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("✓ CUDA optimizations enabled")
    
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
    
    print("\nStarting streaming indexing...")
    print("=" * 60)
    
    # Stream process the PDF with batched embedding and storage
    total_chunks = process_pdf_streaming(pdf_path, collection, embedding_model, device)
    
    print("=" * 60)
    print(f"\n✓ Indexing complete!")
    print(f"✓ Stored {total_chunks} chunks in {CHROMA_DB_PATH}/{COLLECTION_NAME}")


def main():
    """
    Main function to orchestrate the indexing process.
    """
    try:
        print("=" * 60)
        print("AUIS Academic Catalog Indexing (Streaming Mode)")
        print("=" * 60)
        
        # Process PDF in streaming mode with batched writes
        create_chroma_index_streaming(CATALOG_PDF_PATH)
        
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

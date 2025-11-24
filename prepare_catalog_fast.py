"""
AUIS Academic Catalog Indexing Script - Fast Version

This script uses ChromaDB's built-in embedding function for maximum speed.

Usage:
    python prepare_catalog_fast.py
"""

import os

# Disable TensorFlow to avoid Keras compatibility issues
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import re
from typing import List, Dict
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from config import (
    CATALOG_PDF_PATH,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

# Simplified chunking for speed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def chunk_text_simple(text: str) -> List[str]:
    """
    Ultra-fast chunking - just split on sentences.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        if current_length + sentence_len > CHUNK_SIZE and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            # Start new chunk with overlap (last sentence)
            current_chunk = [current_chunk[-1]] if current_chunk else []
            current_length = len(current_chunk[0]) if current_chunk else 0
        
        current_chunk.append(sentence)
        current_length += sentence_len
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def main():
    """
    Fast indexing using ChromaDB's built-in embedding.
    """
    try:
        print("=" * 60)
        print("AUIS Academic Catalog Indexing (FAST MODE)")
        print("=" * 60)
        
        if not os.path.exists(CATALOG_PDF_PATH):
            raise FileNotFoundError(f"PDF not found: {CATALOG_PDF_PATH}")
        
        print(f"\nReading PDF: {CATALOG_PDF_PATH}")
        
        # Step 1: Extract all text and chunk it
        all_chunks = []
        chunk_id = 0
        
        with pdfplumber.open(CATALOG_PDF_PATH) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}\n")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                if not text or len(text.strip()) < 50:
                    continue
                
                # Quick cleanup
                text = ' '.join(text.split())
                
                # Chunk
                page_chunks = chunk_text_simple(text)
                
                for chunk_text in page_chunks:
                    if len(chunk_text) > 50:
                        chunk_id += 1
                        all_chunks.append({
                            'id': f'chunk_{chunk_id}',
                            'text': chunk_text,
                            'metadata': {
                                'page': page_num,
                                'chunk_id': chunk_id
                            }
                        })
                
                if page_num % 20 == 0:
                    print(f"Read {page_num}/{total_pages} pages... ({len(all_chunks)} chunks)")
        
        print(f"\n✓ Extracted {len(all_chunks)} chunks from {total_pages} pages")
        
        # Step 2: Create ChromaDB with built-in embedding
        print(f"\nInitializing ChromaDB with built-in embeddings...")
        
        # Use ChromaDB's default sentence transformer embedding
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Delete old collection
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted old collection: {COLLECTION_NAME}")
        except:
            pass
        
        # Create new collection with embedding function
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
            metadata={"description": "AUIS Academic Catalog"}
        )
        
        print(f"\nAdding {len(all_chunks)} chunks to ChromaDB...")
        print("This will take 5-10 minutes...\n")
        
        # Add in large batches - ChromaDB handles embedding internally
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            ids = [c['id'] for c in batch]
            documents = [c['text'] for c in batch]
            metadatas = [c['metadata'] for c in batch]
            
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            progress = min(i + batch_size, len(all_chunks))
            print(f"Stored {progress}/{len(all_chunks)} chunks...")
        
        print("\n" + "=" * 60)
        print("✓ SUCCESS! Index is ready.")
        print(f"✓ Stored {len(all_chunks)} chunks")
        print("\nYou can now run: python app.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

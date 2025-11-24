"""
Configuration file for AUIS Academic Catalog RAG application.
Adjust these parameters to tune the behavior of the app.
"""

# PDF and data paths
CATALOG_PDF_PATH = "./data/AUIS_Catalog.pdf"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "auis_catalog"

# Chunking parameters
CHUNK_SIZE = 1200  # characters per chunk (larger = fewer chunks = faster)
CHUNK_OVERLAP = 50  # overlap between consecutive chunks (reduced for speed)

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fast 1.1B model for CPU
MAX_NEW_TOKENS = 80  # Decent-length answer, not huge
TEMPERATURE = 0.4  # Balanced for quality and speed
TOP_P = 0.9

# RAG parameters
TOP_K_RETRIEVAL = 1  # number of relevant chunks to retrieve (reduced for speed)

# Gradio UI
APP_TITLE = "AUIS Academic Catalog Assistant"
APP_DESCRIPTION = (
    "Ask questions about AUIS rules, policies, and programs in simple language. "
    "Answers are based on the official academic catalog."
)

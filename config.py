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
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B parameters for speed
MAX_NEW_TOKENS = 256  # Allow complete responses
TEMPERATURE = 0.2  # Lower for more deterministic, focused responses
TOP_P = 0.9

# RAG parameters
TOP_K_RETRIEVAL = 3  # number of relevant chunks to retrieve (increased for better coverage)
SIMILARITY_THRESHOLD = 0.45  # max distance for "good match" (smaller = more similar)

# Gradio UI
APP_TITLE = "AUIS Academic Catalog Assistant"
APP_DESCRIPTION = (
    "Ask questions about AUIS rules, policies, and programs in simple language. "
    "Answers are based on the official academic catalog."
)

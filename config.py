"""
Configuration file for AUIS Academic Catalog RAG application.
Adjust these parameters to tune the behavior of the app.
"""

# PDF and data paths
CATALOG_PDF_PATH = "./data/AUIS_Catalog.pdf"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "auis_catalog"

# Chunking parameters
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9

# RAG parameters
TOP_K_RETRIEVAL = 4  # number of relevant chunks to retrieve

# Gradio UI
APP_TITLE = "AUIS Academic Catalog Assistant"
APP_DESCRIPTION = (
    "Ask questions about AUIS rules, policies, and programs in simple language. "
    "Answers are based on the official academic catalog."
)

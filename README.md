# AUIS Academic Catalog RAG Assistant

A local Retrieval-Augmented Generation (RAG) application that answers questions about the AUIS Academic Catalog using a local LLM running on your GPU.

## Overview

This application enables AUIS students to ask questions about academic policies, rules, and programs in natural language. It uses:

- **Phi-3-mini-4k-instruct** (4-bit quantized) as the local LLM
- **ChromaDB** for vector storage and retrieval
- **sentence-transformers/all-MiniLM-L6-v2** for embeddings
- **pdfplumber** to extract text from the catalog PDF
- **Gradio** for a simple chat interface

Everything runs locally on your machine—no cloud APIs, no fine-tuning, just RAG with a pretrained model.

## Requirements

### System Requirements
- **Python**: 3.10 or later
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4060 with 8GB VRAM)
- **CUDA**: Version 12.4 or compatible

### Python Packages

The following packages are required:

- `torch`, `torchvision`, `torchaudio` (with CUDA 12.4 support)
- `transformers`
- `accelerate`
- `bitsandbytes`
- `chromadb`
- `sentence-transformers`
- `pdfplumber`
- `gradio`

## Installation

### 1. Install PyTorch with CUDA Support

```powershell
python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install Other Dependencies

```powershell
python -m pip install --user transformers accelerate bitsandbytes chromadb sentence-transformers pdfplumber gradio
```

**Note**: Installation may take several minutes as these packages have many dependencies.

## Setup

### 1. Prepare the Catalog PDF

Place your AUIS Academic Catalog PDF at:

```
./data/AUIS_Catalog.pdf
```

The `data/` directory has already been created by the project structure.

### 2. Create the Vector Index

Run the indexing script to process the PDF and create the ChromaDB vector store:

```powershell
python prepare_catalog.py
```

**What this does:**
- Reads all pages from the catalog PDF
- Splits the text into overlapping chunks (~800 characters each)
- Generates embeddings using the MiniLM model
- Stores everything in a persistent ChromaDB collection at `./chroma_db/`

**Expected output:**
```
============================================================
AUIS Academic Catalog Indexing
============================================================
Reading PDF from: ./data/AUIS_Catalog.pdf
Total pages: 150
Processed 10/150 pages...
...
Extracted 450 chunks from 150 pages

Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2
Initializing ChromaDB at: ./chroma_db

Generating embeddings and storing 450 chunks...
Stored 450/450 chunks...

✓ Indexing complete!
✓ Stored 450 chunks in ./chroma_db/auis_catalog

============================================================
SUCCESS! The catalog index is ready.
You can now run: python app.py
============================================================
```

**Note**: This process may take 2-5 minutes depending on the size of your catalog PDF.

## Running the Application

Once the index is created, launch the chat interface:

```powershell
python app.py
```

**What this does:**
- Loads the Phi-3-mini-4k-instruct model in 4-bit quantization on your GPU
- Connects to the ChromaDB collection
- Starts a Gradio web server at `http://127.0.0.1:7860`

**Expected startup output:**
```
============================================================
Initializing Phi-3-mini-4k-instruct model...
============================================================
✓ CUDA available: NVIDIA GeForce RTX 4060
✓ CUDA memory: 8.00 GB

Loading tokenizer from: microsoft/Phi-3-mini-4k-instruct
Loading model in 4-bit quantization...
✓ Model loaded successfully!
============================================================

Connecting to ChromaDB...
✓ Connected to collection: auis_catalog
✓ Collection contains 450 chunks
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
✓ Embedding model loaded

============================================================
Launching Gradio interface...
============================================================

Running on local URL:  http://127.0.0.1:7860
```

**First-time model loading**: The first time you run the app, it will download the Phi-3 model from Hugging Face (~2-3 GB). This is a one-time download and will be cached locally.

### Using the Chat Interface

1. Open your web browser to `http://127.0.0.1:7860`
2. Type your question in the chat box
3. The app will:
   - Retrieve relevant chunks from the catalog
   - Send them to the LLM with your question
   - Generate a student-friendly answer

**Example questions:**
- "What happens if my GPA drops below 2.0?"
- "How many times can I withdraw from a course?"
- "What are the credit hour requirements for graduation?"
- "What is the academic probation policy?"
- "Can I take more than 18 credits in a semester?"

## Project Structure

```
LLM Project/
├── data/
│   └── AUIS_Catalog.pdf          # Place your catalog PDF here
├── chroma_db/                     # Created by prepare_catalog.py
│   └── auis_catalog/              # Vector store collection
├── config.py                      # Configuration parameters
├── prepare_catalog.py             # Indexing script
├── app.py                         # Main RAG application
└── README.md                      # This file
```

## Configuration

You can adjust parameters by editing `config.py`:

- **Chunking**: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Retrieval**: `TOP_K_RETRIEVAL` (how many chunks to retrieve)
- **Generation**: `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`
- **Paths**: `CATALOG_PDF_PATH`, `CHROMA_DB_PATH`

## How It Works

### Indexing Phase (prepare_catalog.py)
1. Extract text from each page of the PDF using pdfplumber
2. Clean and normalize the text
3. Split into overlapping chunks (~800 characters)
4. Generate embeddings for each chunk using MiniLM
5. Store chunks + embeddings in ChromaDB

### Query Phase (app.py)
1. User asks a question
2. Generate embedding for the question
3. Retrieve top-4 most relevant chunks from ChromaDB
4. Build a prompt with:
   - System instructions
   - Retrieved catalog excerpts
   - User's question
5. Send to Phi-3 model for generation
6. Return answer to user

### Key Features
- **Context-aware**: Maintains chat history across multiple turns
- **Source citations**: Mentions page numbers when answering
- **Safety**: System prompt instructs the model to only use catalog content
- **GPU acceleration**: 4-bit quantization allows running on 8GB VRAM
- **No training**: Pure RAG—no fine-tuning or model updates

## Updating the Catalog

When you receive a new version of the catalog:

1. Replace `./data/AUIS_Catalog.pdf` with the new PDF
2. Re-run the indexing script:
   ```powershell
   python prepare_catalog.py
   ```
3. Restart the app

The old index will be deleted and replaced with a fresh one.

## Troubleshooting

### "CUDA not available"
- Verify your NVIDIA drivers are installed
- Check that PyTorch was installed with CUDA support:
  ```powershell
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- Reinstall PyTorch with the correct CUDA version

### "ChromaDB not found"
- Make sure you ran `python prepare_catalog.py` before running `app.py`
- Check that `./chroma_db/` exists in your project directory

### "PDF file not found"
- Verify the catalog PDF is at `./data/AUIS_Catalog.pdf`
- Check the path in `config.py` if you placed it elsewhere

### Out of Memory Errors
- Close other GPU-intensive applications
- Reduce `MAX_NEW_TOKENS` in `config.py`
- Reduce `TOP_K_RETRIEVAL` to retrieve fewer chunks

### Slow Generation
- The first generation is slower as the model warms up
- If consistently slow, check GPU utilization with `nvidia-smi`
- Ensure the model loaded on GPU (check startup messages)

## Notes

- **No fine-tuning**: This is a zero-shot RAG system. The model is not trained on AUIS data.
- **Accuracy**: Answers are only as good as the catalog content retrieved. If a question isn't covered in the catalog, the model will say so.
- **Privacy**: Everything runs locally—no data is sent to external servers.
- **Model cache**: Hugging Face models are cached in `~/.cache/huggingface/`

## License

This project is for educational purposes at AUIS. The AUIS Academic Catalog content remains property of the American University of Iraq, Sulaimani.

## Support

For technical issues with the application, contact your course instructor.

For questions about actual AUIS policies, contact:
- **Registrar's Office**: registrar@auis.edu.krd
- **Academic Advising**: advising@auis.edu.krd

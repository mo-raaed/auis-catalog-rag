# Flask API Backend for AUIS Catalog Assistant

This is the backend API for the AUIS Academic Catalog Assistant. It connects the TinyLlama RAG pipeline with the React frontend built in Lovable.

## Setup

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Make sure you have Flask and Flask-CORS installed:
```bash
python -m pip install flask flask-cors
```

### 2. Run the API Server

```bash
python api_app.py
```

The API will start on `http://127.0.0.1:7864`

## API Endpoints

### `POST /api/ask`

Ask a question about AUIS academic policies.

**Request Body:**
```json
{
  "question": "What is the academic probation policy?",
  "history": [
    ["previous question", "previous answer"]
  ]
}
```

**Response:**
```json
{
  "answer": "The academic probation policy states...",
  "history": [
    ["previous question", "previous answer"],
    ["current question", "current answer"]
  ]
}
```

### `GET /api/health`

Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "AUIS Catalog API is running"
}
```

## Connecting with React Frontend

### In the Lovable React Project:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Set the backend URL in `.env`:
   ```
   VITE_BACKEND_URL=http://127.0.0.1:7864
   ```

3. Install dependencies and run:
   ```bash
   npm install
   npm run dev
   ```

4. Open the Vite dev server URL (usually `http://localhost:5173`)

## CORS Configuration

The API allows requests from:
- `http://localhost:5173` (Vite dev server)
- `http://127.0.0.1:5173`
- `http://localhost:8080`
- `http://127.0.0.1:8080`

If your frontend runs on a different port, update the `CORS` origins in `api_app.py`.

## Performance

- Model initialization: ~15-20 seconds (includes warm-up generation)
- Answer generation: ~2 seconds per question (with GPU)
- ChromaDB retrieval: <100ms

## Architecture

1. **Frontend** (React/Lovable) → Sends question + chat history
2. **Flask API** (`/api/ask`) → Receives request
3. **RAG Pipeline** (`answer_question`) → 
   - Embeds question with sentence-transformers
   - Retrieves relevant chunks from ChromaDB
   - Generates answer with TinyLlama 1.1B on GPU
4. **Response** → Returns answer + updated history to frontend

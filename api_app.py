"""
Small Flask API for the AUIS Academic Catalog Assistant.

This exposes a single endpoint:
POST /api/ask

The React frontend (built in Lovable) will call this endpoint.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your existing RAG functions
from app import initialize_model, initialize_chroma, answer_question

app = Flask(__name__)

# Allow requests from the React dev server
# Adjust ports if your Vite dev server uses a different one.
CORS(app, origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
])

print("Initializing model + Chroma...")
initialize_model()
initialize_chroma()
print("Initialization done. API is ready.")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Expected JSON:
        {
          "question": "string",
          "history": [ [user, assistant], ... ]
        }

    Returns:
        {
          "answer": "string",
          "history": [ [user, assistant], ... ]
        }
    """
    data = request.get_json(force=True) or {}

    question = (data.get("question") or "").strip()
    history = data.get("history") or []

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not isinstance(history, list):
        history = []

    # Call your existing RAG pipeline
    answer = answer_question(question, history)

    # Append this turn to history so the frontend can keep it
    history.append([question, answer])

    return jsonify({"answer": answer, "history": history})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "message": "AUIS Catalog API is running"})


if __name__ == "__main__":
    # Port 7864 matches what Lovable suggested – keep it unless you changed it.
    app.run(host="127.0.0.1", port=7864, debug=False)

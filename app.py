"""
AUIS Academic Catalog RAG Application

This application provides a chat interface for students to ask questions
about AUIS academic policies using Retrieval-Augmented Generation (RAG)
with a local LLM (Phi-3-mini-4k-instruct).

Usage:
    python app.py
"""

import os

# Disable TensorFlow to avoid Keras compatibility issues
# We only need PyTorch for this project
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import time
import torch
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import chromadb
from sentence_transformers import SentenceTransformer
import gradio as gr
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K_RETRIEVAL,
    APP_TITLE,
    APP_DESCRIPTION
)


# Global variables for model and database
tokenizer = None
model = None
chroma_collection = None
embedding_model = None


def initialize_model():
    """
    Load TinyLlama-1.1B-Chat model for fast CPU inference.
    """
    global tokenizer, model
    
    print("\n" + "=" * 60)
    print("Initializing Phi-3-mini-4k-instruct model...")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ Running on CPU (optimized for TinyLlama)")
    
    print(f"\nLoading tokenizer from: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True
    )
    
    # Ensure we have a pad token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model ({'float16 for GPU' if device == 'cuda' else 'float32 for CPU'})...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Move model to device
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully on {device.upper()}!")
    print("=" * 60 + "\n")
    
    # Warm-up: run a tiny generation so CUDA kernels are compiled before first user question
    try:
        print("Running warm-up generation...")
        dummy_inputs = tokenizer(
            "Warm up Phi-3 for AUIS catalog assistant.",
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            _ = model.generate(
                input_ids=dummy_inputs["input_ids"],
                attention_mask=dummy_inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False  # Disable cache for Phi-3
            )
        print("Warm-up generation done.\n")
    except Exception as e:
        print(f"Warm-up failed (non-fatal): {e}\n")


def initialize_chroma():
    """
    Connect to the existing ChromaDB collection.
    """
    global chroma_collection, embedding_model
    
    print("Connecting to ChromaDB...")
    
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"ChromaDB not found at: {CHROMA_DB_PATH}\n"
            f"Please run 'python prepare_catalog.py' first to create the index."
        )
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get the collection
    try:
        chroma_collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Connected to collection: {COLLECTION_NAME}")
        print(f"✓ Collection contains {chroma_collection.count()} chunks")
    except:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found.\n"
            f"Please run 'python prepare_catalog.py' first."
        )
    
    # Load embedding model for query encoding
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("✓ Embedding model loaded\n")


def retrieve_context(query: str, k: int = TOP_K_RETRIEVAL) -> Dict:
    """
    Retrieve top-k relevant chunks from ChromaDB for the given query.
    
    Args:
        query: The user's question
        k: Number of chunks to retrieve
    
    Returns:
        Dictionary with 'combined_context' (str) and 'sources' (list of metadata)
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    # Query ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    
    # Extract documents and metadata
    documents = results['documents'][0]  # List of chunk texts
    metadatas = results['metadatas'][0]  # List of metadata dicts
    
    # Build combined context string with source citations
    context_parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        section = meta.get('section_title', 'Catalog')
        page = meta.get('page', 'N/A')
        context_parts.append(
            f"[Source {i} — {section}, page {page}]\n{doc}"
        )
    
    combined_context = "\n\n".join(context_parts)
    
    return {
        'combined_context': combined_context,
        'sources': metadatas
    }


def generate_llm_response(
    system_prompt: str,
    conversation: List[Dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """
    Generate a response from Phi-3 using plain-text prompting with truncation.
    
    Args:
        system_prompt: System instructions for the model
        conversation: List of {"role": "user"|"assistant", "content": str}
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        The model's response as a string
    """
    # Get only the last user turn (which contains the RAG context + question)
    last_user_content = conversation[-1]["content"] if conversation else ""
    
    # Build a simple plain-text prompt
    prompt_text = f"{system_prompt}\n\n{last_user_content}\n\nAnswer in clear, simple language:\n"
    
    # Tokenize with truncation to keep input manageable
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # keep total input reasonably small
    ).to(model.device)
    
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    
    # Measure generation time
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False  # Disable cache to avoid DynamicCache errors with Phi-3
        )
    end_time = time.time()
    print(f"Generation time in generate_llm_response: {end_time - start_time:.2f} s")
    
    # Decode only the newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    return response


def answer_question(user_message: str, chat_history: List[List[str]]) -> str:
    """
    Main RAG function: retrieve context and generate answer.
    
    Args:
        user_message: Current user question
        chat_history: Gradio chat history (list of [user, assistant] pairs)
    
    Returns:
        The assistant's answer
    """
    print("\n=== New question ===")
    print("User:", user_message)
    
    # Step 1: Retrieve relevant context
    print("Retrieving context...")
    retrieval_result = retrieve_context(user_message, k=TOP_K_RETRIEVAL)
    print("Context retrieved.")
    combined_context = retrieval_result['combined_context']
    
    # Step 2: Define system prompt
    system_prompt = """You are an AI assistant helping students at the American University of Iraq, Sulaimani (AUIS).
You are given excerpts from the official AUIS Academic Catalog.

Your tasks:
- Use ONLY the provided catalog excerpts to answer questions.
- Explain the policies in clear, simple language suitable for undergraduate students.
- Preserve important details like GPA thresholds, number of warnings, credit limits, and consequences.
- When possible, mention which page or section the information came from.
- If the catalog does not clearly answer the question, say so and suggest the student contact the Registrar or Academic Advising for confirmation.
- Do NOT invent new AUIS policies or guess about things not shown in the catalog context."""
    
    # Step 3: Build conversation history
    conversation = []
    
    # Add past turns from chat history
    for user_msg, assistant_msg in chat_history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    
    # Step 4: Add current turn with context
    current_user_content = f"""CONTEXT (excerpts from the AUIS Academic Catalog):

{combined_context}

STUDENT QUESTION:
{user_message}

Using only the context above, answer the student's question in simple, clear language.
If something is missing from the context, say that you don't know and suggest they contact the university for clarification."""
    
    conversation.append({"role": "user", "content": current_user_content})
    
    # Step 5: Generate response with conversation history
    # Include full conversation history for multi-turn context
    # Truncate if needed to stay within token limit
    conversation_for_llm = conversation
    
    # Count approximate tokens (rough estimate: 1 token ≈ 4 chars)
    total_chars = sum(len(msg["content"]) for msg in conversation_for_llm)
    approx_tokens = total_chars // 4
    max_history_tokens = 3000  # Leave room for system prompt + generation
    
    # If conversation is too long, keep only recent turns
    if approx_tokens > max_history_tokens:
        print(f"Conversation too long (~{approx_tokens} tokens), truncating to recent turns...")
        # Keep system context in current turn, drop older turns
        conversation_for_llm = [conversation[-1]]  # Just current turn with context
    
    print(f"Sending {len(conversation_for_llm)} turns to LLM...")
    print("Calling generate_llm_response...")
    try:
        answer = generate_llm_response(system_prompt, conversation_for_llm, max_new_tokens=MAX_NEW_TOKENS)
        print("Generation done.")
        return answer
    except Exception as e:
        print("=" * 80)
        print("GENERATION ERROR:")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"


def qa_single_turn(message: str) -> str:
    """
    Simple wrapper for single-turn Q&A without chat history.
    Useful for a stable Gradio Interface.
    """
    return answer_question(message, [])


def gradio_chat_fn(message: str, history: List[List[str]]) -> str:
    """
    Gradio chat function wrapper.
    
    Args:
        message: Current user message
        history: Chat history (list of [user, assistant] pairs)
    
    Returns:
        Assistant's response
    """
    return answer_question(message, history)


def main():
    """
    Initialize the application and launch Gradio interface.
    """
    try:
        # Initialize model and database
        initialize_model()
        initialize_chroma()
        
        # Create Gradio interface (simple single-turn)
        print("=" * 60)
        print("Launching Gradio interface (single-turn)...")
        print("=" * 60 + "\n")
        
        demo = gr.Interface(
            fn=qa_single_turn,
            inputs=gr.Textbox(label="Your question about AUIS policies"),
            outputs=gr.Textbox(label="Answer"),
            title=APP_TITLE,
            description=APP_DESCRIPTION,
            examples=[
                ["What happens if my GPA drops below 2.0?"],
                ["How many times can I withdraw from a course?"],
                ["What are the credit hour requirements for graduation?"],
                ["What is the academic probation policy?"],
                ["Can I take more than 18 credits in a semester?"]
            ]
        )
        
        # Launch the app
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7864,
            inbrowser=True
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run 'python prepare_catalog.py' first to create the index.")
        
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

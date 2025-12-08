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
import re
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


def clean_answer(text: str) -> str:
    """
    Post-process the model output to remove any leaked control markers.
    """
    # Drop any line that starts with '==='
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("==="):
            continue
        if "<CATALOG>" in line or "</CATALOG>" in line:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    
    # Also remove any stray fragments like '=== CATALOG EXCER'
    cleaned = re.sub(r"===\s*CATALOG.*", "", cleaned)
    
    return cleaned.strip()


def initialize_model():
    """
    Load TinyLlama-1.1B-Chat model.
    """
    global tokenizer, model
    
    print("\n" + "=" * 60)
    print(f"Initializing {LLM_MODEL_NAME}...")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    
    # Load tokenizer
    print(f"Loading tokenizer from: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    
    # Ensure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print(f"Loading model...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Print diagnostics
    print(f"Model device (first param): {next(model.parameters()).device}")
    print(f"Model dtype (first param): {next(model.parameters()).dtype}")
    print(f"✓ Model loaded successfully!")
    print("=" * 60 + "\n")


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
    Generate a response using plain-text prompting with truncation.
    
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
        max_length=1024
    )
    
    # Move to model device
    inputs = inputs.to(model.device)
    
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"Generate using device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")
    
    # Measure generation time with high precision
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.perf_counter()
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
    system_prompt = """You are a friendly AI assistant helping students at the American University of Iraq, Sulaimani (AUIS).

You have two modes of operation:

1) General chat:
   - If the student asks about everyday topics (life, study tips, feelings, etc.), chat naturally and be helpful.
   - You can use your general knowledge in this case.

2) AUIS catalog questions:
   - If the question is about AUIS rules, policies, academic standing, GPA, warnings, credits, majors, minors, or graduation requirements, you MUST use the AUIS Academic Catalog excerpts provided to you.
   - Explain the rules in clear, simple language.
   - Preserve important details (GPA thresholds, number of warnings, credit limits, consequences, deadlines, etc.).
   - Answer in your own words. Do not copy long sentences from the catalog.
   - Never invent new AUIS rules. If the excerpts do NOT contain enough information, say you are not completely sure and suggest contacting the Registrar or Academic Advising for confirmation."""
    
    # Step 3: Build conversation history
    conversation = []
    
    # Add past turns from chat history
    for user_msg, assistant_msg in chat_history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})
    
    # Step 4: Add current turn with stricter catalog-only prompt
    current_user_content = f"""You are answering a question about AUIS.

Here are the only texts you are allowed to use about AUIS policies and programs:

<CATALOG>
{combined_context}
</CATALOG>

Student question:
{user_message}

CRITICAL RULES:

1. Treat everything inside <CATALOG> as the **only source of truth** about AUIS.
2. If the student asks for a **list** (for example: all minors, all majors, all requirements):
   - You may list **only the items that literally appear in <CATALOG>**.
   - If you are not sure the list is complete, say:
     "From the excerpts I can see the following: ... For a complete and up-to-date list, please check the full Academic Catalog or contact the Registrar."
3. Never invent AUIS programs, minors, rules, or numbers that are **not clearly present** in <CATALOG>, even if they sound typical for other universities.
4. Do **not** mention the words "catalog", "context", or "<CATALOG>" in your answer.
5. If <CATALOG> does not give enough information to answer confidently, say that you are not completely sure and recommend that the student check with the Registrar or Academic Advising.

Now write a clear, student-friendly answer to the question above following these rules."""
    
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
        # Keep last 3 turns to maintain some context
        conversation_for_llm = conversation[-3:] if len(conversation) >= 3 else conversation
    
    print(f"Sending {len(conversation_for_llm)} turns to LLM...")
    print("Calling generate_llm_response...")
    try:
        answer = generate_llm_response(system_prompt, conversation_for_llm, max_new_tokens=MAX_NEW_TOKENS)
        
        # Apply comprehensive cleanup to remove leaked markers
        answer = clean_answer(answer)
        
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

import time
from app import initialize_model, initialize_chroma, answer_question

def main():
    print("Initializing model and Chroma...")
    initialize_model()
    initialize_chroma()

    question = "What is the academic probation policy?"
    print("\n=== Direct debug call to answer_question ===")
    print("Question:", question)

    start = time.time()
    answer = answer_question(question, [])
    end = time.time()

    print("\nAnswer:")
    print(answer)
    print(f"\nTotal answer_question time: {end - start:.2f} s")

if __name__ == "__main__":
    main()

from flask import Flask, render_template_string, request
from app import initialize_model, initialize_chroma, answer_question

# Initialize once at startup
print("Initializing AUIS Catalog Assistant (TinyLlama + RAG)...")
initialize_model()
initialize_chroma()
print("Initialization done. Starting web server...")

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>AUIS Academic Catalog Assistant</title>
    <style>
      body { font-family: system-ui, sans-serif; background: #111827; color: #e5e7eb; padding: 2rem; }
      .container { max-width: 900px; margin: 0 auto; }
      h1 { text-align: center; margin-bottom: 0.5rem; }
      p.desc { text-align: center; margin-bottom: 2rem; color: #9ca3af; }
      textarea, input[type=text] {
        width: 100%; padding: 0.75rem; border-radius: 0.5rem;
        border: 1px solid #4b5563; background: #020617; color: #e5e7eb;
      }
      button {
        margin-top: 1rem; padding: 0.75rem 1.5rem; border: none; border-radius: 0.5rem;
        background: #f97316; color: white; font-weight: 600; cursor: pointer;
      }
      button:hover { background: #ea580c; }
      .answer-box {
        margin-top: 1.5rem; padding: 1rem; border-radius: 0.5rem;
        background: #020617; border: 1px solid #4b5563; white-space: pre-wrap;
      }
      label { font-weight: 600; margin-bottom: 0.25rem; display: block; }
      .examples {
        margin-top: 2rem; padding: 1rem; border-radius: 0.5rem;
        background: #1f2937; border: 1px solid #374151;
      }
      .examples h3 { margin-top: 0; color: #f97316; }
      .examples ul { margin: 0; padding-left: 1.5rem; }
      .examples li { margin: 0.5rem 0; color: #9ca3af; cursor: pointer; }
      .examples li:hover { color: #e5e7eb; }
    </style>
    <script>
      function fillExample(text) {
        document.getElementById('question').value = text;
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h1>AUIS Academic Catalog Assistant</h1>
      <p class="desc">
        Ask questions about AUIS rules, policies, and programs in simple language.
        Answers are based on the official academic catalog.
      </p>

      <form method="post">
        <label for="question">Your question about AUIS policies</label>
        <textarea id="question" name="question" rows="3"
          placeholder="What happens if my GPA drops below 2.0?">{{ question }}</textarea>
        <button type="submit">Submit</button>
      </form>

      {% if answer %}
      <div class="answer-box">
        <strong>Answer:</strong><br><br>
        {{ answer }}
      </div>
      {% endif %}

      <div class="examples">
        <h3>Example Questions</h3>
        <ul>
          <li onclick="fillExample('What happens if my GPA drops below 2.0?')">What happens if my GPA drops below 2.0?</li>
          <li onclick="fillExample('How many times can I withdraw from a course?')">How many times can I withdraw from a course?</li>
          <li onclick="fillExample('What are the credit hour requirements for graduation?')">What are the credit hour requirements for graduation?</li>
          <li onclick="fillExample('What is the academic probation policy?')">What is the academic probation policy?</li>
          <li onclick="fillExample('Can I take more than 18 credits in a semester?')">Can I take more than 18 credits in a semester?</li>
        </ul>
      </div>
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            # single-turn: no chat history
            answer = answer_question(question, [])
    return render_template_string(HTML, question=question, answer=answer)

if __name__ == "__main__":
    # Run on the same port you were using for Gradio
    app.run(host="127.0.0.1", port=7864, debug=False)

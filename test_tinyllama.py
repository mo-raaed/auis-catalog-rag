import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda available:", torch.cuda.is_available())
print("device:", device)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.to(device)
model.eval()
t1 = time.time()
print(f"Load time: {t1 - t0:.2f} s")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "In one short sentence, say hello to an AUIS student."}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)

print("Generating...")
g0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
g1 = time.time()

generated = outputs[0][input_ids.shape[1]:]
text = tokenizer.decode(generated, skip_special_tokens=True)

print("Generation time:", f"{g1 - g0:.2f} s")
print("Output:", text)

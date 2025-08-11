from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and dataset
print("Loading FAISS index and dataset...")
index = faiss.read_index("faiss_index.idx")
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def search_similar(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [dataset[i] for i in indices[0]]
    return results

def generate_answer(contexts, query, max_length=100):
    prompt = "Context: " + " ".join(contexts) + "\nQuestion: " + query + "\nAnswer:"

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    generated_tokens = output_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if '\n' in answer:
        answer = answer.split('\n')[0].strip()

    return answer

if __name__ == "__main__":
    print("Mini ChatGPT ready. Type your question or 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        contexts = search_similar(user_query)
        answer = generate_answer(contexts, user_query)
        print("Bot:", answer)

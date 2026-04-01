from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Initialize model and tokenizer
model_name = "gpt2"  # or any GPT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Sample words from different categories
words = [
    # Common words
    "the", "and", "but",
    # Technology
    "computer", "algorithm", "data",
    # Actions
    "run", "think", "learn",
    # Concepts
    "intelligence", "knowledge", "wisdom",
    # Adjectives
    "smart", "fast", "complex",
    # Related terms
    "neural", "network", "brain"
]

# Get embeddings for each word
inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    # Use mean of last hidden states as embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

# Print tokenization information
print("\nTokenization details:")
for word in words:
    tokens = tokenizer.tokenize(word)
    print(f"'{word}' → {tokens}")

print(f"\nGenerated {len(words)} word embeddings of shape {embeddings.shape}")
print("Each embedding is a {}-dimensional vector".format(embeddings.shape[1]))
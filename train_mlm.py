import torch
import torch.nn as nn
import os
from encoder import TransformerEncoder

# -------------------------------
# Setup
# -------------------------------
os.makedirs("results", exist_ok=True)

sentences = [
    "Transformers use [MASK] attention",
    "Mars is called the [MASK] planet",
    "Online learning improves [MASK] access",
    "Exercise improves [MASK] health",
    "Cricket is a [MASK] sport",
    "Python is a [MASK] language",
    "Neural networks have [MASK] layers",
    "Trees reduce [MASK] pollution",
    "Robots perform [MASK] tasks",
    "Solar power is a [MASK] source"
]

targets = [
    "self", "red", "educational", "mental", "popular",
    "programming", "hidden", "air", "repetitive", "renewable"
]

# -------------------------------
# Build Vocabulary
# -------------------------------
vocab = set()
for s in sentences:
    vocab.update(s.split())
vocab.update(targets)
vocab.update(["[PAD]"])

vocab = list(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# -------------------------------
# Encode Dataset
# -------------------------------
inputs = []
labels = []

max_len = max(len(s.split()) for s in sentences)
pad_id = word2idx["[PAD]"]

for s, t in zip(sentences, targets):
    words = s.split()
    input_ids = [word2idx[w] for w in words]
    label_ids = [-100] * len(words)

    mask_index = words.index("[MASK]")
    label_ids[mask_index] = word2idx[t]

    # Padding
    while len(input_ids) < max_len:
        input_ids.append(pad_id)
        label_ids.append(-100)

    inputs.append(input_ids)
    labels.append(label_ids)

inputs = torch.tensor(inputs)
labels = torch.tensor(labels)


# -------------------------------
# Model
# -------------------------------
model = TransformerEncoder(len(vocab))
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training
# -------------------------------
for epoch in range(500):
    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = criterion(logits.view(-1, len(vocab)), labels.view(-1))
    loss.backward()
    optimizer.step()

print("Training completed.")

# -------------------------------
# USER INPUT
# -------------------------------
user_sentence = input("\nEnter a sentence with [MASK]: ")

words = user_sentence.split()
input_ids = torch.tensor([[word2idx.get(w, word2idx["[PAD]"]) for w in words]])

logits, attn = model(input_ids)
mask_index = words.index("[MASK]")

pred_id = logits[0, mask_index].argmax().item()
pred_word = idx2word[pred_id]

words[mask_index] = pred_word
output_sentence = " ".join(words)

# -------------------------------
# Output
# -------------------------------
print("\nReconstructed Sentence:")
print(output_sentence)

with open("results/user_output.txt", "a") as f:
    f.write(f"Input : {user_sentence}\n")
    f.write(f"Output: {output_sentence}\n")

torch.save(attn, "results/user_attention.pt")

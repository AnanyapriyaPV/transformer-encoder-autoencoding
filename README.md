# Transformer Encoder Autoencoding (Mini Masked Language Model)

This project implements a **minimal Transformer Encoder from scratch using PyTorch** to perform **Masked Language Modeling (MLM)** and visualize **Self-Attention weights**.

The implementation demonstrates the core mechanisms used in modern NLP models such as **BERT and GPT**, including:

- Multi-Head Self Attention
- Positional Encoding
- Transformer Encoder Block
- Masked Language Modeling
- Attention Heatmap Visualization

---

# Project Structure

```text
project/
│
├── attention.py                # Multi-head self-attention implementation
├── positional_encoding.py      # Sinusoidal positional encoding
├── encoder.py                  # Transformer encoder architecture
├── train_mlm.py                # MLM training + prediction script
├── visualize_attention.ipynb   # Notebook to visualize attention weights
│
├── results/
│   ├── user_output.txt         # Stores reconstructed sentences
│   ├── user_attention.pt       # Saved attention weights
│   └── attention_heatmap.png   # Generated heatmap visualization
│
└── README.md
```

---

# Transformer Encoder Architecture

The model implements a **single-layer Transformer Encoder block** consisting of:

1. Token Embedding
2. Positional Encoding
3. Multi-Head Self Attention
4. Residual Connection + Layer Normalization
5. Feed Forward Network
6. Masked Language Modeling Head

```
Input Tokens
      ↓
Embedding Layer
      ↓
Positional Encoding
      ↓
Multi-Head Self Attention
      ↓
Add + LayerNorm
      ↓
Feed Forward Network
      ↓
Add + LayerNorm
      ↓
Masked Token Prediction
```

---

# File Descriptions

### `attention.py`
Implements **Multi-Head Self Attention**, which allows each token in a sequence to attend to all other tokens.

### `positional_encoding.py`
Implements **sinusoidal positional encoding** to inject positional information into token embeddings.

### `encoder.py`
Defines the **Transformer Encoder architecture** including:

- Multi-head attention
- Feed-forward layers
- Residual connections
- Layer normalization

### `train_mlm.py`
Main training script that:

- Builds the vocabulary
- Encodes sentences
- Trains the Transformer encoder
- Accepts user input
- Predicts masked tokens
- Saves attention weights

### `visualize_attention.ipynb`
Loads saved attention weights and generates a **self-attention heatmap**.

---

# Example Masked Language Task

Example input:

```
Python is a [MASK] language
```

Model prediction:

```
Python is a programming language
```

---

# Self-Attention Visualization

The heatmap below shows how tokens attend to each other inside the Transformer.

![Self Attention Heatmap](results/attention_heatmap.png)

Each cell represents **attention strength between tokens**.

- X-axis → Key tokens  
- Y-axis → Query tokens  

Brighter colors indicate stronger attention.

---

# Example Output

```
Enter a sentence with [MASK]:

Solar power is a [MASK] source

Reconstructed Sentence:
Solar power is a renewable source
```

---

# Requirements

Install required packages:

```bash
pip install torch matplotlib
```

---

# Running the Project

### 1. Train the Model

Run the training script:

```bash
python train_mlm.py
```

You will be prompted to enter a sentence containing `[MASK]`.

---

### 2️. Visualize Attention

Open the notebook:

```
visualize_attention.ipynb
```

Run the cells to generate the **self-attention heatmap**.

---

# Saving the Heatmap Image

Add this line to the visualization script to save the image:

```python
plt.savefig("results/attention_heatmap.png")
```

This allows the heatmap to appear in the README.

---

# Learning Objectives

This project demonstrates:

- Implementation of **Transformer architecture from scratch**
- Understanding of **self-attention mechanisms**
- Masked language modeling (MLM)
- Visualization of **attention patterns**

---

# Possible Improvements

Future improvements could include:

- Multiple Transformer encoder layers
- Larger training datasets
- Tokenizer integration
- Training on real NLP corpora
- Interactive attention visualization

---

# Author

Educational project demonstrating the **core concepts of Transformer-based NLP models**.
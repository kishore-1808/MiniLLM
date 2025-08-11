# Mini ChatGPT

A simple retrieval-augmented chatbot built using sentence embeddings, FAISS similarity search, and GPT-2 text generation.

## Overview

This project demonstrates a lightweight ChatGPT-like system that:

- Uses a **custom dataset** of factual sentences.
- Encodes the dataset into vector embeddings using **Sentence Transformers**.
- Uses **FAISS** to perform fast similarity search for relevant context based on user query.
- Feeds retrieved context and user query as prompt to **GPT-2** for generating concise answers.

It is ideal as a starting point for building domain-specific Q&A chatbots or proof-of-concept conversational AI systems.

---

## Features

- Efficient semantic search over your dataset using vector embeddings.
- Context-aware generation using pretrained GPT-2.
- Easy to customize dataset and scale.
- Interactive command-line chat interface.

---

## Requirements

- Python 3.7+
- PyTorch
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `pickle`

Install dependencies with:

```bash
pip install torch transformers sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Large dataset of factual sentences
dataset = [
    "Python is a popular programming language used for web development, data science, artificial intelligence, and automation.",
    "JavaScript is mainly used for interactive web development on the client side.",
    "Java is a versatile, object-oriented programming language used for building enterprise applications.",
    "C++ is a powerful programming language commonly used for system/software development and game programming.",
    "Machine learning is a subset of artificial intelligence focused on developing algorithms that learn from data.",
    "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
    "A neural network is a series of algorithms designed to recognize patterns, loosely inspired by the human brain.",
    "Transformers are a deep learning architecture primarily used in natural language processing.",
    "The attention mechanism in transformers allows models to focus on relevant parts of the input.",
    "GPT stands for Generative Pre-trained Transformer and is designed to generate human-like text.",
    "ChatGPT is an AI chatbot developed by OpenAI based on the GPT architecture.",
    "Natural Language Processing, or NLP, is the field of study focused on the interaction between computers and human language.",
    "BERT is a transformer-based model designed for understanding the context of words in text.",
    "Sentence Transformers create embeddings that represent sentences for semantic similarity tasks.",
    "FAISS is a library for efficient similarity search of dense vectors, commonly used in retrieval systems.",
    "Embedding refers to representing words or sentences as dense vectors that capture their meanings.",
    "Fine-tuning is the process of training a pretrained model on a specific dataset to improve performance on a particular task.",
    "Supervised learning uses labeled data to train machine learning models.",
    "Unsupervised learning finds patterns in data without explicit labels.",
    "Reinforcement learning teaches agents to make decisions by rewarding desired behaviors.",
    "Overfitting happens when a model learns the training data too well and fails to generalize.",
    "Regularization techniques help prevent overfitting in machine learning models.",
    "Epochs are the number of times a learning algorithm works through the entire training dataset.",
    "Batch size defines how many samples are processed before the model updates.",
    "Gradient descent is an optimization algorithm to minimize the loss function in training.",
    "Loss function measures how well a machine learning model performs on training data.",
    "Accuracy is the proportion of correct predictions made by a model.",
    "Precision and recall are metrics used to evaluate classification models.",
    "Cross-validation is a technique to assess how well a model generalizes to unseen data.",
    "Tokenization is the process of splitting text into words or subwords.",
    "The vocabulary size of a model defines the number of unique tokens it can understand.",
    "Pretraining involves training a model on a large dataset before fine-tuning it on a specific task.",
    "Zero-shot learning allows a model to perform tasks it was not explicitly trained for.",
    "Few-shot learning enables a model to learn from a small number of examples.",
    "OpenAI develops advanced AI models such as GPT series and Codex.",
    "Codex is a descendant of GPT specialized for understanding and generating code.",
    "Language models predict the next word in a sentence given the previous words.",
    "Sequence-to-sequence models are used for tasks like translation and summarization.",
    "Beam search is a decoding strategy to generate more likely sequences in language models.",
    "Self-attention computes the relationship of each word with every other word in a sequence.",
    "Positional encoding helps transformers understand the order of words.",
    "Dropout is a regularization technique to prevent overfitting by randomly dropping neurons during training.",
    "The transformer architecture was introduced in the paper 'Attention Is All You Need'.",
    "GPT-3 is a large language model with 175 billion parameters.",
    "Training large models requires powerful GPUs or TPUs and large datasets.",
    "Ethics in AI involves ensuring models are fair, transparent, and do not propagate bias.",
    "Explainability helps humans understand how AI models make decisions.",
    "Data preprocessing includes cleaning and transforming raw data into a usable format.",
    "Hyperparameters are model parameters set before training that affect learning.",
    "Optimization algorithms include Adam, RMSProp, and SGD.",
    "Activation functions introduce non-linearity in neural networks, common ones are ReLU and sigmoid.",
    "Backpropagation is the method used to compute gradients for training neural networks.",
    "Convolutional Neural Networks are specialized for image data.",
    "Recurrent Neural Networks are designed to handle sequential data."
]

# Load sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding dataset... This might take a moment.")
corpus_embeddings = embedder.encode(dataset, convert_to_numpy=True)

dimension = corpus_embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Save index and dataset
faiss.write_index(index, "faiss_index.idx")

with open("dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Index and dataset saved successfully!")

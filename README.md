# NLP Preprocessing and Embedding Pipelines

This repository contains two Jupyter notebooks demonstrating text preprocessing and feature extraction for Natural Language Processing (NLP) tasks. It includes two different approaches to transforming raw text into numerical representations:

1. **TF-IDF representation (statistical approach)**  
2. **Word2Vec embeddings (semantic approach)**

---

## Repository Contents

### **1. `tfidf_word_representation.ipynb`**  
This notebook processes raw text and converts it into a **TF-IDF feature matrix**.  
**Main steps:**
- Cleans text by removing unwanted tokens, punctuation, numbers, and special characters.
- Tokenizes words and applies **lemmatization**.
- Removes common English stopwords.
- Optionally splits text into chunks for large datasets.
- Generates a **TF-IDF matrix** using `scikit-learn`’s `TfidfVectorizer`.
- (Optional) Computes **cosine similarity** between text segments.

**Purpose:** Produces document-level feature vectors based on word importance, ideal for tasks such as document similarity, keyword extraction, or clustering.

---

### **2. `word2vec_word_representation.ipynb`**  
This notebook trains a **Word2Vec model** to create dense vector embeddings that capture the semantic meaning of words.  
**Main steps:**
- Performs the same cleaning, tokenization, and lemmatization as the TF-IDF notebook.
- Prepares text into tokenized chunks for efficient model training.
- Trains a Word2Vec model using **Gensim**, with options for:
  - **Skip-Gram (sg=1)** – predicts context words.
  - **CBOW (sg=0)** – predicts a target word from surrounding context.
- Produces **semantic word embeddings** that can be used for word similarity or analogy tasks.

**Purpose:** Creates word-level embeddings that capture semantic relationships and can be used in tasks like semantic search or deep learning.

---

## **TF-IDF vs Word2Vec**
- **TF-IDF:** Focuses on word frequency and uniqueness across documents, without understanding context.  
- **Word2Vec:** Learns vector representations of words that encode semantic meaning and context.

---

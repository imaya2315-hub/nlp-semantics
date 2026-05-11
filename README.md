# Semantic Search Engine using NLP

## Overview

This project implements a semantic search engine that retrieves relevant documents based on meaning rather than exact keyword matching.

It compares:

* TF-IDF (keyword-based search)
* Transformer embeddings (semantic search)
* FAISS (fast vector search)

## Dataset

* AG News dataset (real-world news articles)
* Used 5000+ documents for experimentation

## Features

* Text preprocessing and cleaning
* TF-IDF baseline model
* Semantic search using Sentence-BERT
* Fast similarity search using FAISS
* Evaluation using Precision@K
* Comparison between keyword and semantic retrieval

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Sentence Transformers
* FAISS

## Results

* Semantic search provides more contextually relevant results than TF-IDF
* FAISS significantly improves retrieval speed
* Achieved Precision@3 up to 1.0 on tech-related queries

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the script:
   python src/search.py

## Example Query

Query: "machine learning models"

Semantic Results:

* Relevant AI and technology-related articles retrieved

## Key Learnings

* Difference between lexical and semantic search
* Importance of embeddings in modern search systems
* Efficient similarity search using FAISS

## Future Improvements

* Add web interface (Streamlit)
* Use larger datasets
* Improve evaluation metrics (Recall, MAP)

## Author

Your Name

import pandas as pd
import numpy as np
import re

df = pd.read_csv(r"C:\Users\91861\OneDrive\Documents\Downloads\archive\train.csv")


df['text'] = df['Title'] + " " + df['Description']

documents = df['text'].tolist()[:5000]
labels = df['Class Index'].tolist()[:5000]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2)
)

tfidf_matrix = vectorizer.fit_transform(documents)

def search_tfidf(query):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)
    return scores.flatten()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True
)

def search_semantic(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)
    return scores.flatten()

import faiss

doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

def search_faiss(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    scores, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def get_top_results(scores, documents, top_k=3):
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in ranked_indices]

def print_results(results):
    for i, r in enumerate(results):
        print(f"{i+1}. {r[:120]}\n")

def precision_at_k_category(query, retrieved_docs, k):
    relevant_docs = [
        documents[i] for i in range(len(documents))
        if labels[i] == 4   # tech category
    ]
    
    retrieved_k = retrieved_docs[:k]
    relevant_count = len(set(retrieved_k) & set(relevant_docs))
    return relevant_count / k

queries = [
    "machine learning models",
    "artificial intelligence systems",
    "software technology",
    "data science applications"
]

for query in queries:
    print("\nQuery:", query)

    tfidf_scores = search_tfidf(query)
    semantic_scores = search_semantic(query)

    print("\nTF-IDF Results")
    print_results(get_top_results(tfidf_scores, documents))

    print("Semantic Results")
    print_results(get_top_results(semantic_scores, documents))

    print("FAISS Results")
    print_results(search_faiss(query))

    retrieved_docs = get_top_results(semantic_scores, documents)
    print("Precision@3:", precision_at_k_category(query, retrieved_docs, 3))

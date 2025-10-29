# cli_tester.py

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

CACHE_FILE = "resume_embeddings.pkl"

if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError("Embeddings not found.")

# Load cache
print("Load stored embedds")
with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)

resume_embeddings = cache["embeddings"]
model_name = cache["model_name"]
RESUME_CSV = cache["resume_csv"]

resumes_df = pd.read_csv(RESUME_CSV)

# Combine texts (for keyword scoring)
resume_texts = (
    resumes_df['summary'].fillna('') + " " +
    resumes_df['skills'].fillna('') + " " +
    resumes_df['experience'].fillna('') + " " +
    resumes_df['education'].fillna('')
).tolist()

model = SentenceTransformer(model_name)
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# Hybrid similarity function
def compute_hybrid_similarity(query, top_n=3, keyword_weight=0.2):
    query_embedding = model.encode([query])
    semantic_sim = cosine_similarity(query_embedding, resume_embeddings)[0]

    # Keyword overlap
    query_keywords = [kw[0].lower() for kw in kw_model.extract_keywords(query, top_n=5)]
    keyword_scores = []
    for text in resume_texts:
        score = sum(kw in text.lower() for kw in query_keywords) / len(query_keywords)
        keyword_scores.append(score)

    hybrid_scores = 0.8 * semantic_sim + keyword_weight * np.array(keyword_scores)
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    return top_indices, hybrid_scores

# cli test 
print("\nRizzume Semantic Resume Matcher — CLI Tester")
print("Type your candidate or job description below.")
print("Enter 'exit' to quit.\n")

while True:
    query = input(" Enter candidate requirement/job description: ").strip()
    if query.lower() == "exit":
        print("Exiting.")
        break

    top_indices, scores = compute_hybrid_similarity(query, top_n=3)

    print("\nTop 3 Matching Resumes:==")
    for rank, idx in enumerate(top_indices, start=1):
        r = resumes_df.iloc[idx]
        print(f"#{rank} | 👤 {r['name']}  ({r['domain']})")
        print(f"{r['email']}")
        print(f"Summary: {r['summary']}")
        print(f"Skills: {r['skills']}")
        print(f"Experience: {r['experience']}")
        print(f"Education: {r['education']}")
        print(f"Similarity Score: {scores[idx]:.4f}")
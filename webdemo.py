from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import os

app = Flask(__name__)

# Load data
CACHE_FILE = "rizzume_embeddings.pkl"

# if not os.path.exists(CACHE_FILE):
#     raise FileNotFoundError("Embeddings not found.")

with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)

resume_embeddings = cache["embeddings"]
model_name = cache["model_name"]
RESUME_CSV = cache["resume_csv"]

resumes_df = pd.read_csv(RESUME_CSV)

resume_texts = (
    resumes_df["summary"].fillna("") + " " +
    resumes_df["skills"].fillna("") + " " +
    resumes_df["experience"].fillna("") + " " +
    resumes_df["education"].fillna("")
).tolist()

# Load model
model = SentenceTransformer(model_name)
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# Hybrid similarity 
def compute_hybrid_similarity(query, top_n=3, keyword_weight=0.2):
    query_embedding = model.encode([query])
    semantic_sim = cosine_similarity(query_embedding, resume_embeddings)[0]

    query_keywords = [kw[0].lower() for kw in kw_model.extract_keywords(query, top_n=5)]
    keyword_scores = []
    for text in resume_texts:
        score = sum(kw in text.lower() for kw in query_keywords) / len(query_keywords)
        keyword_scores.append(score)

    hybrid_scores = 0.8 * semantic_sim + keyword_weight * np.array(keyword_scores)
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    return top_indices, hybrid_scores


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ FLASK route \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query_text = ""

    if request.method == "POST":
        query_text = request.form["query"]
        if query_text.strip():
            top_indices, scores = compute_hybrid_similarity(query_text, top_n=3)
            for idx in top_indices:
                r = resumes_df.iloc[idx]
                results.append({
                    "name": r["name"],
                    "domain": r["domain"],
                    "email": r["email"],
                    "summary": r["summary"],
                    "skills": r["skills"],
                    "experience": r["experience"],
                    "education": r["education"],
                    "score": round(float(scores[idx]), 4)
                })

    return render_template("index.html", query=query_text, results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
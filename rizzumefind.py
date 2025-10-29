import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

RESUME_CSV = "resumes.csv"
CACHE_FILE = "rizzume_embeddings.pkl"

print("loading")
resumes_df = pd.read_csv(RESUME_CSV)

resume_texts = (
    resumes_df['summary'].fillna('') + " " +
    resumes_df['skills'].fillna('') + " " +
    resumes_df['experience'].fillna('') + " " +
    resumes_df['education'].fillna('')
).tolist()

print("Sentence-BERT ")
model = SentenceTransformer('all-mpnet-base-v2')

print("embedding")
rizzume_embeddings = model.encode(resume_texts, show_progress_bar=True)
with open(CACHE_FILE, "wb") as f:
    pickle.dump({
        "embeddings": rizzume_embeddings,
        "model_name": "all-mpnet-base-v2",
        "resume_csv": RESUME_CSV
    }, f)

print("Embeddings  cached :", CACHE_FILE)
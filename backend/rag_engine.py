# === RAG Engine: Handles Embedding + Retrieval ===
from sentence_transformers import SentenceTransformer, util
import pandas as pd

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_dataframe(self, df, text_column='chunk'):
        if text_column not in df.columns:
            raise ValueError("Column 'chunk' not found in CSV.")
        self.df = df.copy()
        self.df['embedding'] = self.df[text_column].astype(str).apply(
            lambda x: self.model.encode(x, convert_to_tensor=True)
        )
        return self.df

    def query(self, question, top_k=3):
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        self.df['similarity'] = self.df['embedding'].apply(
            lambda emb: util.cos_sim(emb, question_embedding).item()
        )
        top_chunks = self.df.sort_values(by='similarity', ascending=False).head(top_k)
        return top_chunks['chunk'].tolist()

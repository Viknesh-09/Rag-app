# === RAG Engine: Handles Embedding + Retrieval + Generation ===
# Import necessary libraries
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os # Required for accessing environment variables (like API keys)
from groq import Groq # New: Import the Groq client library

class RAGEngine:
    # Constructor for RAGEngine
    def __init__(self):
        # Initialize the Sentence Transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Groq client for LLM generation
        # This line securely fetches the GROQ_API_KEY from environment variables (Streamlit Secrets)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Method to embed content from a DataFrame
    def embed_dataframe(self, df, text_column='chunk'):
        # Check if the required 'chunk' column exists
        if text_column not in df.columns:
            raise ValueError("Column 'chunk' not found in CSV.")
        self.df = df.copy()
        # Generate embeddings for each chunk in the specified column
        self.df['embedding'] = self.df[text_column].astype(str).apply(
            lambda x: self.model.encode(x, convert_to_tensor=True)
        )
        return self.df

    # Method to generate an answer using the Groq LLM
    def _generate_answer_with_llm(self, question, retrieved_chunks):
        # Combine retrieved chunks into a single context string
        context = "\n".join(retrieved_chunks)

        # Define the prompt messages for the LLM
        # System message sets the LLM's persona
        # User message provides the context and the question
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, state that you don't have enough information."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]

        try:
            # Make the API call to Groq for chat completions
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192", # Specifies the Groq model to use (e.g., Llama 3 8B)
                temperature=0.7, # Controls creativity (0.0 for deterministic, 1.0 for more creative)
                max_tokens=500, # Sets the maximum length of the generated answer
            )
            # Extract and return the generated content from the LLM's response
            return chat_completion.choices[0].message.content
        except Exception as e:
            # Handle potential errors during LLM generation
            print(f"Error generating answer with LLM: {e}")
            return f"Error: Could not generate an answer (Details: {e}). Please check your Groq API key and model settings."

    # Main method to query the RAG system
    def query(self, question, top_k=3):
        # Encode the user's question into an embedding
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        # Calculate similarity between question and all chunk embeddings
        self.df['similarity'] = self.df['embedding'].apply(
            lambda emb: util.cos_sim(emb, question_embedding).item()
        )
        # Retrieve the top K most similar chunks
        top_chunks = self.df.sort_values(by='similarity', ascending=False).head(top_k)
        retrieved_chunks = top_chunks['chunk'].tolist()

        # Use the retrieved chunks to generate a final answer with the LLM
        generated_answer = self._generate_answer_with_llm(question, retrieved_chunks)

        # Return the generated answer as a list (to fit app.py's expected format)
        return [generated_answer]

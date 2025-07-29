# === Streamlit Frontend ===
import streamlit as st
import pandas as pd
from backend.rag_engine import RAGEngine
from backend.memory import ChatMemory

st.set_page_config(page_title="RAG Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 RAG Chunking Analyzer</h1>", unsafe_allow_html=True)

rag_engine = RAGEngine()
chat_memory = ChatMemory()

uploaded_file = st.file_uploader("📤 Upload your CSV (must include a 'chunk' column):", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        rag_engine.embed_dataframe(df)
        st.success("✅ Embeddings generated successfully.")
    except ValueError as e:
        st.error(str(e))

    st.markdown("### 💬 Ask a question from your document:")
    user_question = st.text_input("Type your question here:")

    if st.button("Ask"):
        if user_question:
            results = rag_engine.query(user_question)
            chat_memory.add_message("User", user_question)
            chat_memory.add_message("RAG", "\n".join(results))

            st.markdown("#### 🧠 Answer")
            st.success("\n\n".join(results))
        else:
            st.warning("Please enter a question.")

    st.markdown("### 🗃️ Chat Memory")
    for entry in chat_memory.get_history():
        st.markdown(f"**{entry['user']}**: {entry['message']}")

else:
    st.info("Please upload a CSV file with a `chunk` column to begin.")

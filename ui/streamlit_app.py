import requests
import streamlit as st

API_URL = "http://rag-api:8000"  

st.set_page_config(page_title="RAG QA UI", layout="wide")
st.title("OpenRAGBench ArXiv RAG QA")

question = st.text_input("Ask a question about the arXiv corpus")
top_k = st.slider("Top K retrieved chunks", 1, 10, 5)

if st.button("Ask") and question:
    res = requests.post(
        f"{API_URL}/query",   
        json={"question": question, "top_k": top_k},
        timeout=180,
    )
    res.raise_for_status()
    data = res.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Retrieved context")
    for i, c in enumerate(data["context"], start=1):
        st.markdown(
            f"**{i}. {c.get('title','')}**  \n"
            f"Doc: {c['doc_id']} . Section: {c['section_id']}"
        )
        st.text(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))

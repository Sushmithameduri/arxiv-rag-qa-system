from __future__ import annotations
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from app.config import settings


class RAGService:
    def __init__(self):
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBED_MODEL
        )

        # Vector DB
        self.vectordb = Chroma(
            persist_directory=settings.CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name=settings.CHROMA_COLLECTION,
        )

        # Local LLM via Ollama
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_HOST,
            temperature=0.1,
        )

    # -------------------------
    # Retrieval
    # -------------------------
    def retrieve(self, question: str, top_k: int = 5):
        return self.vectordb.similarity_search(question, k=top_k)

    # -------------------------
    # Prompt with citations
    # -------------------------
    def _build_cited_prompt(
        self,
        question: str,
        docs: List[Dict[str, Any]],
    ) -> str:
        context_blocks = []

        for i, d in enumerate(docs, start=1):
            block = (
                f"[{i}] (Doc: {d['doc_id']}, Section: {d['section_id']})\n"
                f"{d['text']}"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
You are an expert research assistant.

Your task is to answer the question by identifying:
1. The core research problem or limitation motivating the paper.
2. Why existing approaches are insufficient.

Use ONLY the provided context.
You may combine multiple context blocks to infer the answer.

If the problem cannot be identified, say:
"I don’t know based on the provided documents."

If the problem is strongly supported by the context, state it confidently.
Avoid speculative language when evidence is clear.

Write a concise, technical answer (3–4 sentences max).
Cite document IDs in brackets.

Question:
{question}

Context:
{context_text}

Answer:
"""
        return prompt.strip()

    # -------------------------
    # RAG Answer with citations
    # -------------------------
    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Retrieve relevant chunks
        docs = self.retrieve(question, top_k=top_k)

        retrieved_docs: List[Dict[str, Any]] = []
        for d in docs:
            retrieved_docs.append(
                {
                    "doc_id": str(d.metadata.get("doc_id", "")),
                    "section_id": int(d.metadata.get("section_id", -1)),
                    "title": d.metadata.get("title"),
                    "text": d.page_content,
                }
            )

        # 2. Build citation-aware prompt
        prompt = self._build_cited_prompt(question, retrieved_docs)

        # 3. Generate answer with Ollama
        response = self.llm.invoke(prompt)

        # 4. Return answer + retrieved context
        return {
            "answer": response,
            "context": retrieved_docs,
        }

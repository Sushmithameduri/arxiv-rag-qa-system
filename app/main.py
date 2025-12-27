from fastapi import FastAPI

from app.schemas import QueryRequest, QueryResponse, RetrieveResponse, ContextItem
from app.rag import RAGService

app = FastAPI(title="OpenRAGBench ArXiv RAG QA API")

rag = RAGService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: QueryRequest):
    docs = rag.retrieve(req.question, top_k=req.top_k)
    context = [
        ContextItem(
            doc_id=str(d.metadata.get("doc_id", "")),
            section_id=int(d.metadata.get("section_id", -1)),
            title=d.metadata.get("title"),
            text=d.page_content,
        )
        for d in docs
    ]
    return RetrieveResponse(context=context)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = rag.answer(req.question, top_k=req.top_k)
    context = [ContextItem(**c) for c in result["context"]]
    return QueryResponse(answer=result["answer"], context=context)

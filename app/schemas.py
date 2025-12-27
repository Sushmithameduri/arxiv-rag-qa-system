from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class ContextItem(BaseModel):
    doc_id: str
    section_id: int
    title: Optional[str] = None
    text: str

class QueryResponse(BaseModel):
    answer: str
    context: List[ContextItem]

class RetrieveResponse(BaseModel):
    context: List[ContextItem]

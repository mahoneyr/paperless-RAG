from typing import Literal, Optional
from pydantic import BaseModel


class Document(BaseModel):
    id: int
    title: str
    content: str
    created_date: Optional[str] = None
    tags: list[str] = []


class SearchRequest(BaseModel):
    question: str
    mode: Literal["fast", "thinking"] = "fast"


class SourceDoc(BaseModel):
    id: int
    title: str


class SearchResult(BaseModel):
    question: str
    search_query: str
    document_count: int
    summary: str
    sources: list[SourceDoc]
    mode: str


class HealthStatus(BaseModel):
    paperless: bool
    ollama: bool
    paperless_error: Optional[str] = None
    ollama_error: Optional[str] = None

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
    model: Optional[str] = None


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


class AnswerRequest(BaseModel):
    question: str
    documents: list[Document]
    model: Optional[str] = None


class SearchIndexRequest(BaseModel):
    document_type: Optional[list[str]] = None
    correspondent: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    search_text: Optional[str] = None


class HealthStatus(BaseModel):
    paperless: bool
    ollama: bool
    paperless_error: Optional[str] = None
    ollama_error: Optional[str] = None

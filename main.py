import logging
import os
import secrets
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.llm import LLMClient
from app.models import (
    HealthStatus, SearchRequest, SearchResult, AnswerRequest, SearchIndexRequest,
    NotebookCreateRequest, NotebookPushRequest, NotebookPushResult,
)
from app.notebook import OpenNotebookClient
from app.orchestrator import SearchAndSummarize
from app.paperless import PaperlessClient
from app.streaming import progress_events

import pathlib
env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

paperless_client: PaperlessClient = None
llm_client: LLMClient = None
orchestrator: SearchAndSummarize = None
notebook_client: OpenNotebookClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global paperless_client, llm_client, orchestrator, notebook_client

    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable is required — authentication is mandatory. "
            "Set API_KEY in your environment (e.g. the Portainer stack) and redeploy."
        )
    app.state.api_key = api_key

    # Optional integration: off unless OPEN_NOTEBOOK_URL is set
    open_notebook_url = os.getenv("OPEN_NOTEBOOK_URL", "").strip()
    if open_notebook_url:
        notebook_client = OpenNotebookClient(open_notebook_url, os.getenv("OPEN_NOTEBOOK_PASSWORD", ""))
        logging.info(f"Open Notebook integration enabled at {open_notebook_url}")
    else:
        logging.info("Open Notebook integration disabled (OPEN_NOTEBOOK_URL not set)")

    paperless_url = os.environ["PAPERLESS_URL"]
    paperless_token = os.environ["PAPERLESS_TOKEN"]
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
    ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    logging.info(f"DEBUG: OLLAMA_MODEL from env = {ollama_model!r}")

    paperless_client = PaperlessClient(paperless_url, paperless_token)
    llm_client = LLMClient(ollama_url, ollama_model, ollama_embed_model)
    orchestrator = SearchAndSummarize(paperless_client, llm_client)

    logging.info(f"Connecting to Paperless at {paperless_url}")
    logging.info(f"Using Ollama model '{ollama_model}' at {ollama_url}")
    logging.info(f"Using embed model '{ollama_embed_model}'")
    orchestrator.load_taxonomy()
    yield


app = FastAPI(title="Paperless LLM Search", lifespan=lifespan)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Gate all /api/* routes behind the X-API-Key header. Static files and the
    UI shell are served freely; the document data is what is protected."""
    if request.url.path.startswith("/api/"):
        provided = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(provided, app.state.api_key):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


def require_notebook_client() -> None:
    if notebook_client is None:
        raise HTTPException(status_code=404, detail="Open Notebook integration is not configured")


@app.get("/api/notebooks")
def list_notebooks():
    require_notebook_client()
    try:
        return notebook_client.list_notebooks()
    except httpx.HTTPError as e:
        logging.warning(f"Open Notebook unreachable: {e}")
        raise HTTPException(status_code=503, detail="Open Notebook is unreachable")


@app.post("/api/notebooks")
def create_notebook(request: NotebookCreateRequest):
    require_notebook_client()
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Notebook name cannot be empty")
    try:
        if any(n["name"].lower() == name.lower() for n in notebook_client.list_notebooks()):
            raise HTTPException(status_code=409, detail=f"A notebook named “{name}” already exists")
        return notebook_client.create_notebook(name, request.description.strip())
    except httpx.HTTPError as e:
        logging.warning(f"Notebook creation failed: {e}")
        raise HTTPException(status_code=502, detail="Open Notebook could not create the notebook")


@app.post("/api/notebooks/push", response_model=list[NotebookPushResult])
def push_to_notebook(request: NotebookPushRequest):
    """Copy the Paperless text of each document into an Open Notebook notebook.
    Content is read fresh from Paperless, so re-OCR'd text is what gets sent."""
    require_notebook_client()
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="No documents selected")
    try:
        existing = notebook_client.source_titles(request.notebook_id)
        transformations = notebook_client.default_transformations()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404,
                                detail="That notebook no longer exists in Open Notebook. Pick another one.")
        logging.warning(f"Open Notebook error: {e}")
        raise HTTPException(status_code=502, detail=f"Open Notebook returned {e.response.status_code}")
    except httpx.HTTPError as e:
        logging.warning(f"Open Notebook unreachable: {e}")
        raise HTTPException(status_code=503, detail="Open Notebook is unreachable")

    results = []
    for doc_id in request.document_ids:
        try:
            doc = paperless_client.get_document_detail(doc_id)
        except httpx.HTTPStatusError as e:
            detail = ("Not found in Paperless" if e.response.status_code == 404
                      else f"Paperless returned {e.response.status_code}")
            results.append(NotebookPushResult(id=doc_id, title=f"#{doc_id}", status="failed", detail=detail))
            continue
        except httpx.HTTPError as e:
            results.append(NotebookPushResult(id=doc_id, title=f"#{doc_id}", status="failed",
                                              detail=f"Paperless unreachable: {e}"))
            continue

        title = " ".join(p for p in (doc["created"], doc["correspondent"], doc["title"]) if p)
        if title in existing:
            results.append(NotebookPushResult(id=doc_id, title=title, status="skipped",
                                              detail="Already in this notebook"))
            continue
        if not doc["content"].strip():
            results.append(NotebookPushResult(id=doc_id, title=title, status="failed",
                                              detail="Document has no text in Paperless"))
            continue

        header = "\n".join(line for line in (
            f"Paperless document #{doc_id}",
            f"Title: {doc['title']}",
            f"Correspondent: {doc['correspondent']}" if doc["correspondent"] else "",
            f"Date: {doc['created']}" if doc["created"] else "",
        ) if line)
        try:
            notebook_client.add_text_source(request.notebook_id, title,
                                            f"{header}\n\n{doc['content']}", transformations)
            existing.add(title)
            results.append(NotebookPushResult(id=doc_id, title=title, status="sent"))
        except httpx.HTTPError as e:
            logging.warning(f"Push of document {doc_id} failed: {e}")
            results.append(NotebookPushResult(id=doc_id, title=title, status="failed",
                                              detail=f"Open Notebook: {e}"))

    logging.info(f"Pushed to {request.notebook_id}: "
                 + ", ".join(f"{r.id}={r.status}" for r in results))
    return results


@app.get("/api/health", response_model=HealthStatus)
def health():
    status = HealthStatus(paperless=False, ollama=False)

    try:
        paperless_client.ping()
        status.paperless = True
    except Exception as e:
        status.paperless_error = str(e)

    try:
        llm_client.ping()
        status.ollama = True
    except Exception as e:
        status.ollama_error = str(e)

    return status


@app.get("/api/filters")
def get_filters():
    """Return available document types, correspondents, and tags for filtering."""
    try:
        taxonomy = orchestrator.taxonomy
        return {
            "paperless_url": os.getenv("PAPERLESS_PUBLIC_URL", os.environ["PAPERLESS_URL"]).rstrip("/"),
            "document_types": [{"id": dt, "name": dt} for dt in taxonomy.get("document_types", [])],
            "correspondents": [{"id": c, "name": c} for c in taxonomy.get("correspondents", [])],
            "tags": [{"id": t, "name": t} for t in taxonomy.get("tags", [])],
            "models": llm_client.get_available_models(),
        }
    except Exception:
        logging.exception("Error fetching filters")
        raise HTTPException(status_code=500, detail="Could not fetch filters")


@app.post("/api/search/index")
def search_index(request: SearchIndexRequest):
    """Search Paperless with filters, return documents, and generate initial summary."""
    try:
        document_type = request.document_type or []
        correspondent = request.correspondent or []
        tags = request.tags or []
        search_text = request.search_text

        # Build search query with AND logic across all filters
        q_parts = []

        # Add document type filters (OR within the same filter type)
        if document_type:
            type_parts = [f'type:"{dt}"' for dt in document_type]
            if len(type_parts) == 1:
                q_parts.append(type_parts[0])
            else:
                q_parts.append(f"({' OR '.join(type_parts)})")

        # Add correspondent filters (OR within the same filter type)
        if correspondent:
            corr_parts = [f'correspondent:"{c}"' for c in correspondent]
            if len(corr_parts) == 1:
                q_parts.append(corr_parts[0])
            else:
                q_parts.append(f"({' OR '.join(corr_parts)})")

        # Add tags filters (OR within the same filter type)
        if tags:
            tag_parts = [f'tags:"{t}"' for t in tags]
            if len(tag_parts) == 1:
                q_parts.append(tag_parts[0])
            else:
                q_parts.append(f"({' OR '.join(tag_parts)})")

        # Add text search
        if search_text:
            q_parts.append(search_text)

        # Combine all parts with AND
        if q_parts:
            query = " AND ".join(q_parts)
        else:
            query = "*"

        logging.info(f"Searching Paperless with query: {query!r}")

        # Execute search
        docs = paperless_client.search(query)
        logging.info(f"Found {len(docs)} documents")

        return {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "created_date": doc.created_date,
                }
                for doc in docs
            ]
        }
    except Exception:
        logging.exception("Error searching documents")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/api/search/answer")
def search_answer(request: AnswerRequest):
    """Answer a question using RAG on the supplied documents."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        logging.info(f"Answering question about {len(request.documents)} documents")
        answer, relevant_docs = orchestrator.answer(request.question, request.documents, model=request.model)
        return {
            "question": request.question,
            "answer": answer,
            "document_count": len(request.documents),
            "relevant_count": len(relevant_docs),
        }
    except Exception:
        logging.exception("Error generating answer")
        raise HTTPException(status_code=500, detail="Could not generate answer")


@app.post("/api/search/answer-stream")
async def search_answer_stream(request: AnswerRequest):
    """Stream progress while answering a question about selected documents."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    def work(progress):
        answer, relevant_docs = orchestrator.answer(
            request.question, request.documents, model=request.model, progress=progress
        )
        return {
            "question": request.question,
            "summary": answer,
            "document_count": len(request.documents),
            "sources": [{"id": doc.id, "title": doc.title} for doc in relevant_docs],
        }

    return StreamingResponse(progress_events(work), media_type="text/event-stream")


@app.post("/api/search", response_model=SearchResult)
def search(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return orchestrator.process(request.question, request.mode, model=request.model)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Unexpected error during search")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def work(progress):
        return orchestrator.process(request.question, request.mode, progress, model=request.model).model_dump()

    return StreamingResponse(progress_events(work), media_type="text/event-stream")


@app.get("/")
async def root():
    """Serve the main UI."""
    # no-cache: browsers revalidate, so a new release shows up on a normal reload
    return FileResponse("static/index.html", media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory="static", html=True), name="static")

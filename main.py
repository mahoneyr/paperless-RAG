import logging
import os
import pathlib
import secrets
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.filters import build_index_queries
from app.llm import LLMClient
from app.models import HealthStatus, SearchRequest, SearchResult, AnswerRequest, SearchIndexRequest
from app.orchestrator import SearchAndSummarize
from app.paperless import PaperlessClient
from app.streaming import progress_events

env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable is required — authentication is mandatory. "
            "Set API_KEY in your environment (e.g. the Portainer stack) and redeploy."
        )
    app.state.api_key = api_key

    paperless_url = os.environ["PAPERLESS_URL"]
    paperless_token = os.environ["PAPERLESS_TOKEN"]
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
    ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    app.state.paperless = PaperlessClient(paperless_url, paperless_token)
    app.state.llm = LLMClient(ollama_url, ollama_model, ollama_embed_model)
    app.state.orchestrator = SearchAndSummarize(app.state.paperless, app.state.llm)

    logging.info(f"Connecting to Paperless at {paperless_url}")
    logging.info(f"Using Ollama model '{ollama_model}' at {ollama_url}")
    logging.info(f"Using embed model '{ollama_embed_model}'")
    app.state.orchestrator.load_taxonomy()
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


@app.get("/api/health", response_model=HealthStatus)
def health():
    status = HealthStatus(paperless=False, ollama=False)

    try:
        app.state.paperless.ping()
        status.paperless = True
    except Exception as e:
        status.paperless_error = str(e)

    try:
        app.state.llm.ping()
        status.ollama = True
    except Exception as e:
        status.ollama_error = str(e)

    return status


@app.get("/api/filters")
def get_filters():
    """Return available document types, correspondents, and tags for filtering."""
    try:
        taxonomy = app.state.orchestrator.taxonomy
        return {
            "paperless_url": os.getenv("PAPERLESS_PUBLIC_URL", os.environ["PAPERLESS_URL"]).rstrip("/"),
            "document_types": [{"id": dt, "name": dt} for dt in taxonomy.get("document_types", [])],
            "correspondents": [{"id": c, "name": c} for c in taxonomy.get("correspondents", [])],
            "tags": [{"id": t, "name": t} for t in taxonomy.get("tags", [])],
            "models": app.state.llm.get_available_models(),
        }
    except Exception:
        logging.exception("Error fetching filters")
        raise HTTPException(status_code=500, detail="Could not fetch filters")


@app.post("/api/search/index")
def search_index(request: SearchIndexRequest):
    """Search Paperless with filters and return matching documents."""
    try:
        queries = build_index_queries(
            request.document_type or [],
            request.correspondent or [],
            request.tags or [],
            request.search_text,
        )
        logging.info(f"Searching Paperless with {len(queries)} queries")

        # Execute all queries and combine results, removing duplicates.
        seen_ids = set()
        all_documents = []
        for q in queries:
            logging.debug(f"Executing query: {q}")
            for doc in app.state.paperless.search(q):
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_documents.append(doc)

        logging.info(f"Found {len(all_documents)} unique documents")

        return {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "created_date": doc.created_date,
                }
                for doc in all_documents
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
        answer, relevant_docs = app.state.orchestrator.answer(
            request.question, request.documents, model=request.model
        )
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
        answer, relevant_docs = app.state.orchestrator.answer(
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
        return app.state.orchestrator.process(request.question, request.mode, model=request.model)
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
        result = app.state.orchestrator.process(
            request.question, request.mode, progress, model=request.model
        )
        return result.model_dump()

    return StreamingResponse(progress_events(work), media_type="text/event-stream")


@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse("static/index.html", media_type="text/html")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

import asyncio
import json
import logging
import os
import queue
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.llm import LLMClient
from app.models import HealthStatus, SearchRequest, SearchResult
from app.orchestrator import SearchAndSummarize
from app.paperless import PaperlessClient

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global paperless_client, llm_client, orchestrator

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
            "document_types": [{"id": dt, "name": dt} for dt in taxonomy.get("document_types", [])],
            "correspondents": [{"id": c, "name": c} for c in taxonomy.get("correspondents", [])],
            "tags": [{"id": t, "name": t} for t in taxonomy.get("tags", [])],
        }
    except Exception:
        logging.exception("Error fetching filters")
        raise HTTPException(status_code=500, detail="Could not fetch filters")


@app.post("/api/search/index")
def search_index(request: dict):
    """Search Paperless with filters, return documents, and generate initial summary."""
    try:
        document_type = request.get("document_type")
        correspondent = request.get("correspondent")
        tags = request.get("tags")
        search_text = request.get("search_text")

        # Build search query from filters
        filters = []
        if document_type:
            filters.append(f'type:"{document_type}"')
        if correspondent:
            filters.append(f'correspondent:"{correspondent}"')
        if tags:
            filters.append(f'tags:"{tags}"')
        if search_text:
            filters.append(search_text)

        query = " ".join(filters) if filters else "*"
        logging.info(f"Searching Paperless with query: {query}")

        documents = paperless_client.search(query)
        logging.info(f"Found {len(documents)} documents")

        return {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                }
                for doc in documents
            ]
        }
    except Exception:
        logging.exception("Error searching documents")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/api/search/answer")
def search_answer(request: dict):
    """Answer a question using RAG on indexed documents."""
    try:
        question = request.get("question")
        documents = request.get("documents", [])

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")

        logging.info(f"Answering question about {len(documents)} documents")

        # Embed question and documents for ranking
        logging.info(f"Embedding question and {len(documents)} documents for ranking")
        question_embedding = llm_client._embed(question)

        # Rank documents by embedding similarity
        import math
        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return dot / (mag_a * mag_b)

        scored = []
        for doc in documents:
            # Embed document if not already embedded
            if "embedding" not in doc:
                doc["embedding"] = llm_client._embed(f"{doc['title']}\n{doc['content']}")
            score = cosine_similarity(question_embedding, doc["embedding"])
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        import os
        top_k = int(os.getenv("MAX_SUMMARY", "5"))
        relevant_docs = [doc for _, doc in scored[:top_k]]
        logging.info(f"Selected {len(relevant_docs)} most relevant documents for answer")

        # Combine relevant documents and answer the question with RAG
        combined_text = "\n\n---\n\n".join(
            f"Document: {doc['title']}\n{doc['content']}" for doc in relevant_docs
        )
        answer = llm_client.rag_answer(combined_text, question)

        return {
            "question": question,
            "answer": answer,
            "document_count": len(documents),
            "relevant_count": len(relevant_docs)
        }
    except Exception:
        logging.exception("Error generating answer")
        raise HTTPException(status_code=500, detail="Could not generate answer")


@app.post("/api/search", response_model=SearchResult)
def search(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return orchestrator.process(request.question, request.mode)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Unexpected error during search")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    progress_queue: queue.SimpleQueue = queue.SimpleQueue()

    def progress(message: str):
        progress_queue.put({"type": "progress", "message": message})

    async def event_generator():
        try:
            logging.info(f"Starting stream for question: {request.question!r}")
            search_task = asyncio.create_task(
                asyncio.to_thread(orchestrator.process, request.question, request.mode, progress)
            )
            logging.info("Search task created")

            # Yield progress messages while task is running
            while not search_task.done():
                while not progress_queue.empty():
                    try:
                        msg = progress_queue.get_nowait()
                        logging.debug(f"Yielding progress: {msg}")
                        yield f"data: {json.dumps(msg)}\n\n"
                    except Exception as e:
                        logging.exception("Error getting progress message from queue")
                        break
                await asyncio.sleep(0.1)

            # Drain any remaining progress messages after task completes
            while not progress_queue.empty():
                try:
                    msg = progress_queue.get_nowait()
                    logging.debug(f"Yielding final progress: {msg}")
                    yield f"data: {json.dumps(msg)}\n\n"
                except Exception as e:
                    logging.exception("Error draining final progress messages")
                    break

            # Get the result - await the task to catch any exceptions it raised
            try:
                result = await search_task
                result_dict = result.model_dump()
                logging.info(f"Search completed, yielding result with {result.document_count} documents")
                yield f"data: {json.dumps({'type': 'result', 'data': result_dict})}\n\n"
            except ConnectionError as e:
                logging.error(f"Connection error during search: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            except Exception as e:
                logging.exception("Unexpected error during streaming search")
                yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred'})}\n\n"
        except Exception as e:
            logging.exception("Unexpected error in event generator")
            yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse("static/index.html", media_type="text/html")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

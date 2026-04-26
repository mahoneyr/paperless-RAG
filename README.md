# Paperless LLM RAG Q&A

A question-answering interface for [Paperless-NGX](https://docs.paperless-ngx.com/) documents using RAG (Retrieval Augmented Generation). Search and filter your documents, then ask questions and get synthesized answers powered by local or cloud LLMs via [Ollama](https://ollama.ai/).

## Features

- **No Pre-Indexing Required** — Unlike other RAG applications, index and ask questions on-the-fly. Select documents dynamically and get answers in real-time without batch processing
- **Document Filtering** — Filter documents by type, correspondent, tags, and text search
- **Document Selection** — Choose which documents to use for Q&A from search results
- **Smart Ranking** — Ranks selected documents by relevance using embedding similarity
- **Ask Questions** — Get synthesized answers from your selected documents using an LLM
- **Web UI** — Clean, responsive interface for searching, selecting, and asking questions
- **Flexible Model Selection** — Use local models via Ollama for full privacy, or cloud models like Gemini for better quality (your choice)

## Workflow

**Phase 1: Search & Filter**
1. User selects filters: document type, correspondent, tags
2. User optionally searches for text within documents to further narrow the list
3. Paperless-NGX returns matching documents
4. User reviews and selects which documents to use for Q&A

**Phase 2: Q&A on Selected Documents**
1. User asks a question
2. Embedding Model (Ollama) ranks selected documents by relevance
3. LLM (Ollama) synthesizes an answer from the ranked documents
4. Answer returned to user

## Requirements

- **Python 3.9+**
- **Paperless-NGX** instance (accessible over network)
- **Ollama** server with LLM and embedding models
  - **Recommended**: `llama3.1` for LLM, `embeddinggemma` for embeddings
  - Other options: `mistral`, `gemini-3-flash-preview:cloud`, etc. (Ollama supports many models)
  - Note: Some cloud models require external API calls; local models process everything on your machine

## Installation

### Option 1: Local Setup

#### 1. Clone the repository
```bash
git clone https://github.com/mahoneyr/paperless-rag.git
cd paperless-rag
```

#### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure environment
Copy `.env.template` to `.env` and fill in your configuration:

```bash
cp .env.template .env
```

Edit `.env`:
```env
# Paperless-NGX
PAPERLESS_URL=http://your-paperless-host:8000
PAPERLESS_TOKEN=your-api-token-here
PAPERLESS_PUBLIC_URL=http://your-paperless-host:8000  # Optional: external URL for document links

# Ollama
OLLAMA_URL=http://your-ollama-host:11434
OLLAMA_MODEL=llama3.1
OLLAMA_EMBED_MODEL=embeddinggemma

# Optional settings
MAX_RESULTS=1000           # Max documents to retrieve per search
MAX_SUMMARY=20             # Max documents to use for answer synthesis
```

### Getting API credentials

**Paperless-NGX Token:**
1. Log in to your Paperless instance
2. Go to Settings → API tokens
3. Generate a new token and copy it to `PAPERLESS_TOKEN`

**Ollama Setup:**
1. Install Ollama from https://ollama.ai/
2. Pull the recommended models:
   ```bash
   ollama pull llama3.1
   ollama pull embeddinggemma
   ```
3. Start the Ollama server (usually runs on localhost:11434)

### Option 2: Docker Deployment

The repository includes a `Dockerfile` for containerized deployment:

```bash
docker build -t paperless-llm .
docker run -p 8000:8000 \
  -e PAPERLESS_URL=http://your-paperless-host:8000 \
  -e PAPERLESS_TOKEN=your-token \
  -e OLLAMA_URL=http://your-ollama-host:11434 \
  -e OLLAMA_MODEL=llama3.1 \
  -e OLLAMA_EMBED_MODEL=embeddinggemma \
  paperless-llm
```

Or use `docker-compose.yml` for a complete stack (requires editing for your environment):

```bash
docker-compose up
```

## Running the Server

```bash
python main.py
```

The server starts on `http://localhost:8000`. Open it in your browser to access the web UI.

## Technical Details

The sections below are for developers and integrators. Most users only need to open `http://localhost:8000` in a browser to get started.

### API Endpoints

### Health Check
```
GET /api/health
```
Returns the status of Paperless and Ollama connections.

### Get Filters
```
GET /api/filters
```
Returns available document types, correspondents, and tags for filtering.

### Search Documents
```
POST /api/search/index
Content-Type: application/json

{
  "document_type": ["Invoice", "Receipt"],
  "correspondent": ["Vendor A"],
  "tags": ["important"],
  "search_text": "keywords"
}
```
Returns indexed documents matching the filters.

### Get Answer
```
POST /api/search/answer
Content-Type: application/json

{
  "question": "What was the total amount spent?",
  "documents": [...]  // Array of documents to analyze
}
```
Returns a synthesized answer based on the selected documents.

### Stream Answer
```
POST /api/search/answer-stream
Content-Type: application/json

{
  "question": "What was the total amount spent?",
  "documents": [...]
}
```
Returns a streaming response of the answer using Server-Sent Events (SSE).

## Development

### Project Structure
```
paperless-llm/
├── main.py                 # FastAPI server and routes
├── app/
│   ├── paperless.py       # Paperless-NGX client
│   ├── llm.py             # Ollama LLM and embedding client
│   ├── orchestrator.py     # Search and answer synthesis logic
│   ├── filters.py         # Filter helper functions
│   └── models.py          # Pydantic models
├── static/
│   └── index.html         # Web UI
├── requirements.txt       # Python dependencies
└── .env.template          # Environment template
```

### Adding a new LLM model

Edit `.env` and set `OLLAMA_MODEL` to your model name, then restart the server.

## Troubleshooting

**"Could not fetch filters"** — Check that Paperless is running and accessible at `PAPERLESS_URL`.

**"Connection refused to Ollama"** — Ensure Ollama is running and `OLLAMA_URL` is correct. Models may take time to load on first use.

**Slow responses** — Larger documents and models take longer. Adjust `MAX_SUMMARY` to use fewer documents, or use a faster model.

## License

MIT License — see LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to open issues and pull requests.

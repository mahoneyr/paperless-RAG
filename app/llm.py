import hashlib
import math
import logging
import os
import threading
import time
from collections import OrderedDict

import httpx

from .chunking import split_text

logger = logging.getLogger(__name__)

MAX_CHARS = 80000       # ~20k tokens: total document text sent to the answering model
DOC_MAX_CHARS = 20000   # per-document cap; longer documents are reduced to their best passages
EMBED_MAX_CHARS = 8000  # safety cap on a single embedding input (passages are far smaller)
EMBED_BATCH = 32        # passages per /api/embed request
PASSAGE_CACHE_DOCS = int(os.getenv("PASSAGE_CACHE_DOCS", "500"))  # documents whose passage embeddings stay in memory

SUMMARIZE_PROMPT = """\
Answer this question using only the documents below. Do not summarize the documents. \
Give a direct, specific answer to the question. Include key facts, dates, and figures.

Question: {question}

Documents:
{documents}

Answer:"""

DOC_SUMMARIZE_PROMPT = """\
Question: {question}

Read the document below and extract only the facts that directly help answer the question above.
Do not summarize the document. If the document contains nothing relevant, reply with "Not relevant."

Document: {title}
{content}

Relevant facts:"""

SYNTHESIZE_PROMPT = """\
Answer this question using only the facts extracted below. Do not summarize. \
Give a direct, specific answer. Include key facts, dates, and figures. \
Ignore any entries marked "Not relevant."

Question: {question}

Facts from {doc_count} documents:
{summaries}

Answer:"""

RAG_PROMPT = """\
Answer this question using only the documents below. Do not summarize the documents. \
Be direct and specific. When you use information from a document, include the document \
title in italics in parentheses at the end of the relevant sentence, like this: (*Document Title*).

Question: {question}

Documents:
{documents}

Answer:"""


def _context_from_passages(content: str, scored: list[tuple[float, int, str]], budget: int) -> str:
    """Whole document if it fits ``budget``; otherwise its best passages in document order."""
    if len(content) <= budget:
        return content
    chosen, used = [], 0
    for score, offset, passage in sorted(scored, key=lambda s: s[0], reverse=True):
        if used + len(passage) > budget:
            continue
        chosen.append((offset, passage))
        used += len(passage)
    chosen.sort()
    return "\n[…]\n".join(p for _, p in chosen)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class LLMClient:
    def __init__(self, base_url: str, model: str, embed_model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self._passage_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()

    def _generate(self, prompt: str, model: str = None) -> str:
        url = f"{self.base_url}/api/generate"
        selected_model = model or self.model
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_ctx": int(os.getenv("OLLAMA_CTX", "131072"))},
        }

        start = time.time()
        for attempt in range(4):
            response = httpx.post(url, json=payload, timeout=300)
            if response.status_code == 403:
                raise ValueError(
                    f"Access denied for model '{selected_model}'. "
                    f"This may be a paid cloud model. Check your Ollama subscription or use a free model."
                )
            if response.status_code == 404:
                raise ValueError(f"Ollama model '{selected_model}' not found")
            if response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Rate limited by Ollama (attempt {attempt + 1}), retrying in {wait}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            response.raise_for_status()

        elapsed = time.time() - start
        result = response.json()["response"].strip()
        logger.info(f"LLM inference took {elapsed:.2f}s (model={selected_model}, prompt_len={len(prompt)})")
        return result

    def _embed(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts, EMBED_BATCH per request."""
        url = f"{self.base_url}/api/embed"
        vectors = []
        with httpx.Client(timeout=120) as client:
            for i in range(0, len(texts), EMBED_BATCH):
                batch = [t[:EMBED_MAX_CHARS] for t in texts[i:i + EMBED_BATCH]]
                response = client.post(url, json={"model": self.embed_model, "input": batch})
                if response.status_code == 404:
                    raise ValueError(f"Ollama embed model '{self.embed_model}' not found — check OLLAMA_EMBED_MODEL in your .env")
                response.raise_for_status()
                vectors.extend(response.json()["embeddings"])
        return vectors

    def _uses_gemma_prompts(self) -> bool:
        # EmbeddingGemma is trained with task prefixes; omitting them lowers retrieval quality
        return "embeddinggemma" in self.embed_model.lower()

    def embed_query(self, question: str) -> list[float]:
        if self._uses_gemma_prompts():
            return self._embed(f"task: search result | query: {question}")
        return self._embed(question)

    def _passage_vectors(self, doc) -> list[tuple[int, str, list[float]]]:
        """(offset, passage, vector) for every passage of a document, cached by content."""
        key = (self.embed_model, doc.id, hashlib.sha1(f"{doc.title}\0{doc.content}".encode()).hexdigest())
        with self._cache_lock:
            if key in self._passage_cache:
                self._passage_cache.move_to_end(key)
                return self._passage_cache[key]

        passages = split_text(doc.content)
        if not passages:  # no text at all: let the title stand in
            passages = [(0, "")]
        if self._uses_gemma_prompts():
            texts = [f"title: {doc.title or 'none'} | text: {p}" for _, p in passages]
        else:
            texts = [f"{doc.title}\n{p}" for _, p in passages]
        vectors = self._embed_many(texts)
        result = [(off, p, v) for (off, p), v in zip(passages, vectors)]

        with self._cache_lock:
            self._passage_cache[key] = result
            while len(self._passage_cache) > PASSAGE_CACHE_DOCS:
                self._passage_cache.popitem(last=False)
        return result

    def select_relevant(self, documents, question: str, progress=None,
                        total_budget: int = MAX_CHARS, top_k: int = None):
        """Rank documents by their best-matching passage and build each top document's context.

        Returns [(doc, score, context_text)] for the top MAX_SUMMARY documents. A document that
        fits its share of the budget is passed whole; a longer one is reduced to its passages
        most similar to the question, kept in document order, so text deep inside it is
        both searchable and visible to the answering model.
        """
        logger.info(f"Embedding and ranking {len(documents)} documents by passage")
        question_vec = self.embed_query(question)
        ranked = []
        total = len(documents)
        for idx, doc in enumerate(documents, 1):
            if progress:
                progress(f"Analyzing {idx}/{total}: {doc.title}")
            try:
                scored = [(_cosine_similarity(question_vec, v), off, p)
                          for off, p, v in self._passage_vectors(doc)]
            except Exception as e:
                logger.warning(f"Failed to embed '{doc.title}': {e} — scoring 0")
                scored = [(0.0, 0, doc.content)]
            ranked.append((max(s for s, _, _ in scored), doc, scored))

        ranked.sort(key=lambda r: r[0], reverse=True)
        top = ranked[:top_k or int(os.getenv("MAX_SUMMARY", "5"))]
        per_doc = min(DOC_MAX_CHARS, total_budget // max(len(top), 1))
        selected = [(doc, score, _context_from_passages(doc.content, scored, per_doc))
                    for score, doc, scored in top]
        logger.info("Top docs: " + ", ".join(f"{d.title} ({s:.3f})" for d, s, _ in selected))
        return selected

    def summarize(self, documents_text: str, question: str, model: str = None) -> str:
        logger.info("Summarizing documents (single-pass)")
        truncated = documents_text[:MAX_CHARS]
        if len(documents_text) > MAX_CHARS:
            logger.warning(f"Document text truncated from {len(documents_text)} to {MAX_CHARS} chars")
            truncated += "\n\n[Note: document text was truncated due to length]"
        return self._generate(SUMMARIZE_PROMPT.format(question=question, documents=truncated), model=model)

    def summarize_document(self, title: str, content: str, question: str, model: str = None) -> str:
        truncated = content[:DOC_MAX_CHARS]
        if len(content) > DOC_MAX_CHARS:
            logger.warning(f"Document '{title}' truncated from {len(content)} to {DOC_MAX_CHARS} chars")
        return self._generate(DOC_SUMMARIZE_PROMPT.format(
            question=question, title=title, content=truncated
        ), model=model)

    def synthesize(self, doc_summaries: list[tuple[str, str]], question: str, model: str = None) -> str:
        logger.info(f"Synthesizing {len(doc_summaries)} document summaries")
        formatted = "\n\n---\n\n".join(
            f"From '{title}':\n{summary}" for title, summary in doc_summaries
        )
        return self._generate(SYNTHESIZE_PROMPT.format(
            question=question, doc_count=len(doc_summaries), summaries=formatted
        ), model=model)

    def rag_answer(self, documents_text: str, question: str, model: str = None) -> str:
        """Answer a question using RAG (Retrieval Augmented Generation)."""
        logger.info("Answering question with RAG")
        return self._generate(RAG_PROMPT.format(question=question, documents=documents_text), model=model)

    def ping(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(str(e))

    def get_available_models(self) -> list[dict]:
        """Fetch available models from Ollama with size information."""
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                models = []
                for model in data.get("models", []):
                    models.append({
                        "name": model["name"],
                        "size": model.get("size", 0),
                    })
                return sorted(models, key=lambda x: x["name"])
        except Exception as e:
            logger.error(f"Failed to fetch models from Ollama: {e}")
            return []

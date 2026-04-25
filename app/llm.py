import math
import logging
import os
import time
import httpx

logger = logging.getLogger(__name__)

MAX_CHARS = 80000       # ~20k tokens for single-pass summarization
DOC_MAX_CHARS = 20000   # per-document cap for map-reduce individual summaries
EMBED_MAX_CHARS = 8000  # truncation for embedding input

TRANSLATE_PROMPT = "Extract ONLY proper nouns, names, or company/provider types. Return just the search term(s), minimal words:\n\nQ: Have I received bills from Liberty Mutual?\nA: Liberty Mutual\n\nQ: What utilities do I pay?\nA: utilities\n\nQ: {question}\nA:"

SUMMARIZE_PROMPT = """\
The user asked: "{question}"

Based on the following document(s), provide a concise summary that directly answers the question.
Include key facts, dates, and figures where relevant.

{documents}

Summary:"""

DOC_SUMMARIZE_PROMPT = """\
The user asked: "{question}"

Extract only the information from the document below that is relevant to answering that question.
Be concise. If the document contains nothing relevant, reply with "Not relevant."

Document: {title}
{content}

Relevant information:"""

SYNTHESIZE_PROMPT = """\
The user asked: "{question}"

Below are summaries extracted from {doc_count} documents. Synthesize them into a single, \
cohesive answer. Include key facts, dates, and figures. Ignore any summaries marked "Not relevant."

{summaries}

Final answer:"""

RAG_PROMPT = """\
Answer the question based ONLY on the provided documents. Be direct and concise.

Question: {question}

Documents:
{documents}

Answer:"""


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

    def _generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        start = time.time()
        for attempt in range(4):
            response = httpx.post(url, json=payload, timeout=300)
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
        logger.info(f"LLM inference took {elapsed:.2f}s (model={self.model}, prompt_len={len(prompt)})")
        return result

    def _embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.embed_model, "input": text[:EMBED_MAX_CHARS]}
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()["embeddings"][0]

    def translate_query(self, question: str) -> str:
        logger.info(f"Translating question to search query: {question!r}")
        result = self._generate(TRANSLATE_PROMPT.format(question=question))
        logger.info(f"Translated query: {result!r}")
        return result or question

    def rank_documents(self, documents, question: str):
        logger.info(f"Embedding and ranking {len(documents)} documents")
        question_vec = self._embed(question)
        scored = []
        for doc in documents:
            try:
                doc_vec = self._embed(f"{doc.title}\n{doc.content}")
                score = _cosine_similarity(question_vec, doc_vec)
            except Exception as e:
                logger.warning(f"Failed to embed '{doc.title}': {e} — scoring 0")
                score = 0.0
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = int(os.getenv("MAX_SUMMARY", "5"))
        top = [doc for _, doc in scored[:top_k]]
        logger.info(f"Top {len(top)} docs: {[d.title for d in top]}")
        return top

    def summarize(self, documents_text: str, question: str) -> str:
        logger.info("Summarizing documents (single-pass)")
        truncated = documents_text[:MAX_CHARS]
        if len(documents_text) > MAX_CHARS:
            logger.warning(f"Document text truncated from {len(documents_text)} to {MAX_CHARS} chars")
            truncated += "\n\n[Note: document text was truncated due to length]"
        return self._generate(SUMMARIZE_PROMPT.format(question=question, documents=truncated))

    def summarize_document(self, title: str, content: str, question: str) -> str:
        truncated = content[:DOC_MAX_CHARS]
        if len(content) > DOC_MAX_CHARS:
            logger.warning(f"Document '{title}' truncated from {len(content)} to {DOC_MAX_CHARS} chars")
        return self._generate(DOC_SUMMARIZE_PROMPT.format(
            question=question, title=title, content=truncated
        ))

    def synthesize(self, doc_summaries: list[tuple[str, str]], question: str) -> str:
        logger.info(f"Synthesizing {len(doc_summaries)} document summaries")
        formatted = "\n\n---\n\n".join(
            f"From '{title}':\n{summary}" for title, summary in doc_summaries
        )
        return self._generate(SYNTHESIZE_PROMPT.format(
            question=question, doc_count=len(doc_summaries), summaries=formatted
        ))

    def rag_answer(self, documents_text: str, question: str) -> str:
        """Answer a question using RAG (Retrieval Augmented Generation)."""
        logger.info("Answering question with RAG")
        return self._generate(RAG_PROMPT.format(question=question, documents=documents_text))

    def ping(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(str(e))

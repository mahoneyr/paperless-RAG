import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .paperless import PaperlessClient
from .llm import LLMClient
from .models import SearchResult, SourceDoc
from .filters import extract_filters, build_filter_string

logger = logging.getLogger(__name__)

Progress = Optional[Callable[[str], None]]

STOP_WORDS = {
    'what', 'when', 'where', 'who', 'why', 'how', 'do', 'did', 'does', 'have', 'has', 'had',
    'is', 'are', 'am', 'be', 'been', 'being', 'the', 'a', 'an', 'and', 'or', 'but', 'not',
    'in', 'on', 'at', 'by', 'for', 'to', 'of', 'from', 'with', 'as', 'if', 'about', 'into',
    'up', 'out', 'so', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'any',
    'some', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'my', 'your',
    'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those', 'i', 'you',
    'he', 'she', 'it', 'we', 'they', 'been', 'being', 'received', 'get', 'got'
}

def extract_keywords(question: str) -> str:
    """Extract keywords from question by removing stop words and cleaning punctuation."""
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[?!.,;:]', ' ', question.lower())
    # Split into words
    words = cleaned.split()
    # Keep words that aren't stop words and are longer than 1 char
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(keywords) if keywords else question


class SearchAndSummarize:
    def __init__(self, paperless: PaperlessClient, llm: LLMClient):
        self.paperless = paperless
        self.llm = llm
        self.taxonomy: dict = {}

    def load_taxonomy(self):
        self.taxonomy = self.paperless.get_taxonomy()
        logger.info(f"Loaded taxonomy: {len(self.taxonomy['tags'])} tags, "
                    f"{len(self.taxonomy['correspondents'])} correspondents, "
                    f"{len(self.taxonomy['document_types'])} document types")

    def process(self, question: str, mode: str = "fast", progress: Progress = None) -> SearchResult:
        logger.info(f"Processing question: {question!r} (mode={mode})")

        _progress(progress, "Building search query...")
        keywords = extract_keywords(question)

        filters = extract_filters(question, self.taxonomy)
        filter_string = build_filter_string(filters)
        search_query = f"{keywords} {filter_string}".strip()
        _progress(progress, f"Search query: {search_query!r}")
        logger.info(f"Final search query: {search_query!r}")
        _progress(progress, f"Searching Paperless: {search_query!r}")

        documents = self.paperless.search(search_query)

        if not documents:
            return SearchResult(
                question=question,
                search_query=search_query,
                document_count=0,
                summary="No documents were found matching your question.",
                sources=[],
                mode=mode,
            )

        _progress(progress, f"Found {len(documents)} documents")

        if mode == "thinking":
            summary, used_docs = self._thinking(question, documents, progress)
        else:
            summary, used_docs = self._fast(question, documents, progress)

        return SearchResult(
            question=question,
            search_query=search_query,
            document_count=len(documents),
            summary=summary,
            sources=[SourceDoc(id=doc.id, title=doc.title) for doc in used_docs],
            mode=mode,
        )

    def _fast(self, question: str, documents, progress: Progress) -> tuple[str, list]:
        _progress(progress, f"Ranking {len(documents)} documents by relevance...")
        top_docs = self.llm.rank_documents(documents, question)
        _progress(progress, f"Summarizing top {len(top_docs)} documents...")
        combined = "\n\n---\n\n".join(
            f"Document: {doc.title} (ID: {doc.id})\n{doc.content}"
            for doc in top_docs
        )
        return self.llm.summarize(combined, question), top_docs

    def _thinking(self, question: str, documents, progress: Progress) -> tuple[str, list]:
        total = len(documents)
        doc_summaries: list[tuple[str, str]] = [None] * total

        def summarize_one(index: int, doc):
            _progress(progress, f"Analyzing {index + 1}/{total}: {doc.title}")
            summary = self.llm.summarize_document(doc.title, doc.content, question)
            _progress(progress, f"Done {index + 1}/{total}: {doc.title}")
            return index, doc.title, summary

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(summarize_one, i, doc) for i, doc in enumerate(documents)]
            for future in as_completed(futures):
                index, title, summary = future.result()
                doc_summaries[index] = (title, summary)

        _progress(progress, "Synthesizing final answer...")
        return self.llm.synthesize(doc_summaries, question), documents


def _progress(callback: Progress, message: str):
    logger.info(message)
    if callback:
        callback(message)

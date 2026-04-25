import logging
import os
import httpx
from .models import Document

logger = logging.getLogger(__name__)


class PaperlessClient:
    def __init__(self, base_url: str, token: str):
        self.max_results = int(os.getenv("MAX_RESULTS", "20"))
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Token {token}"}

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(headers=self.headers, timeout=30) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def search(self, query: str) -> list[Document]:
        logger.info(f"Searching Paperless: {query!r}")
        results = []
        params = {"query": query, "page_size": self.max_results}

        data = self._get("/api/documents/", params=params)
        for item in data.get("results", []):
            results.append(Document(
                id=item["id"],
                title=item["title"],
                content=item.get("content", ""),
                created_date=item.get("created"),
                tags=[str(t) for t in item.get("tags", [])],
            ))

        logger.info(f"Found {len(results)} documents")
        return results

    def get_document(self, doc_id: int) -> str:
        data = self._get(f"/api/documents/{doc_id}/")
        return data.get("content", "")

    def _get_names(self, path: str) -> list[str]:
        names = []
        params = {"page_size": 500}
        while True:
            data = self._get(path, params=params)
            names.extend(item["name"] for item in data.get("results", []))
            if not data.get("next"):
                break
            params["page"] = params.get("page", 1) + 1
        return sorted(names)

    def get_taxonomy(self) -> dict:
        logger.info("Fetching Paperless taxonomy (tags, correspondents, document types)")
        return {
            "tags": self._get_names("/api/tags/"),
            "correspondents": self._get_names("/api/correspondents/"),
            "document_types": self._get_names("/api/document_types/"),
        }

    def ping(self) -> bool:
        try:
            self._get("/api/documents/", params={"page_size": 1})
            return True
        except Exception as e:
            raise ConnectionError(str(e))

import logging
import httpx

logger = logging.getLogger(__name__)


class OpenNotebookClient:
    """Minimal client for the Open Notebook REST API."""

    def __init__(self, base_url: str, password: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {password}"} if password else {}

    def _request(self, method: str, path: str, **kwargs):
        with httpx.Client(headers=self.headers, timeout=60) as client:
            response = client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            return response.json()

    def list_notebooks(self) -> list[dict]:
        data = self._request("GET", "/api/notebooks", params={"archived": False})
        return [{"id": n["id"], "name": n["name"]} for n in data]

    def create_notebook(self, name: str, description: str = "") -> dict:
        n = self._request("POST", "/api/notebooks", json={"name": name, "description": description})
        return {"id": n["id"], "name": n["name"]}

    def source_titles(self, notebook_id: str) -> set[str]:
        """Titles of every source already in the notebook, used to skip duplicates."""
        titles, offset, page = set(), 0, 100
        while True:
            batch = self._request("GET", "/api/sources", params={
                "notebook_id": notebook_id, "limit": page, "offset": offset,
            })
            titles.update(s.get("title") or "" for s in batch)
            if len(batch) < page:
                return titles
            offset += page

    def default_transformations(self) -> list[str]:
        data = self._request("GET", "/api/transformations")
        return [t["id"] for t in data if t.get("apply_default")]

    def add_text_source(self, notebook_id: str, title: str, content: str,
                        transformations: list[str]) -> dict:
        return self._request("POST", "/api/sources/json", json={
            "type": "text",
            "notebooks": [notebook_id],
            "title": title,
            "content": content,
            "transformations": transformations,
            "embed": True,
            "async_processing": True,
        })

    def ping(self) -> bool:
        self._request("GET", "/api/notebooks", params={"archived": False})
        return True

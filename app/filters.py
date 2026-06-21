import re
from datetime import date
from typing import Optional


def build_index_queries(
    document_type: list[str],
    correspondent: list[str],
    tags: list[str],
    search_text: Optional[str],
) -> list[str]:
    """Build Paperless search queries for the index endpoint.

    OR within a single metadata type (separate queries combined by the caller),
    AND across types (cartesian product). Optional free-text is appended to each.
    """
    def with_text(query: str) -> str:
        return f"{query} {search_text}".strip() if search_text else query

    if not (document_type or correspondent or tags):
        return [search_text] if search_text else ["*"]

    # Single metadata type: one query per value (OR semantics).
    if document_type and not correspondent and not tags:
        return [with_text(f'type:"{dt}"') for dt in document_type]
    if correspondent and not document_type and not tags:
        return [with_text(f'correspondent:"{c}"') for c in correspondent]
    if tags and not document_type and not correspondent:
        return [with_text(f'tags:"{t}"') for t in tags]

    # Multiple metadata types: cartesian product (AND across types, OR within).
    queries = []
    for dt in (document_type or [None]):
        for c in (correspondent or [None]):
            for tg in (tags or [None]):
                parts = []
                if dt:
                    parts.append(f'type:"{dt}"')
                if c:
                    parts.append(f'correspondent:"{c}"')
                if tg:
                    parts.append(f'tags:"{tg}"')
                if search_text:
                    parts.append(search_text)
                if parts:
                    queries.append(" ".join(parts))
    return queries


def extract_filters(question: str, taxonomy: dict) -> dict:
    q = question.lower()
    filters = {}

    match = _best_match(q, taxonomy.get("correspondents", []))
    if match:
        filters["correspondent"] = match

    match = _best_match(q, taxonomy.get("document_types", []))
    if match:
        filters["type"] = match

    match = _best_match(q, taxonomy.get("tags", []))
    if match:
        filters["tag"] = match

    filters.update(_extract_date_range(question))
    return filters


def build_filter_string(filters: dict) -> str:
    parts = []
    if "correspondent" in filters:
        parts.append(f'correspondent:"{filters["correspondent"]}"')
    if "type" in filters:
        parts.append(f'type:"{filters["type"]}"')
    if "tag" in filters:
        parts.append(f'tag:"{filters["tag"]}"')
    if "date_gte" in filters:
        parts.append(f'created:>={filters["date_gte"]}')
    if "date_lte" in filters:
        parts.append(f'created:<={filters["date_lte"]}')
    return " ".join(parts)


def _best_match(question: str, names: list[str]) -> Optional[str]:
    matches = [n for n in names if n.lower() in question]
    return max(matches, key=len) if matches else None


def _extract_date_range(question: str) -> dict:
    q = question.lower()
    current_year = date.today().year

    if "last year" in q:
        y = current_year - 1
        return {"date_gte": f"{y}-01-01", "date_lte": f"{y}-12-31"}

    if "this year" in q:
        y = current_year
        return {"date_gte": f"{y}-01-01", "date_lte": f"{y}-12-31"}

    years = sorted(set(re.findall(r'\b(20\d{2})\b', question)))
    if len(years) == 1:
        return {"date_gte": f"{years[0]}-01-01", "date_lte": f"{years[0]}-12-31"}
    if len(years) >= 2:
        return {"date_gte": f"{years[0]}-01-01", "date_lte": f"{years[-1]}-12-31"}

    return {}

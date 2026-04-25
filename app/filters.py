import re
from datetime import date
from typing import Optional


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

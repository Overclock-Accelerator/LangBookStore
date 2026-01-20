"""
Shared helpers for working with `storedata.json`.

Used by tools like GetAnswers / RecommendBooks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_store_books() -> list[dict[str, Any]]:
    path = _project_root() / "storedata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    books = data.get("books", [])
    if not isinstance(books, list):
        return []
    return [b for b in books if isinstance(b, dict)]


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _NON_ALNUM_RE.sub(" ", s.lower()).strip()


def fmt_money(value: Any) -> str | None:
    try:
        return f"${float(value):.2f}"
    except Exception:
        return None


@dataclass(frozen=True)
class BookView:
    id: int | None
    title: str
    author: str
    genre: str
    rating: float | None
    pages: int | None
    price: float | None
    year: int | None
    description: str
    review_count: int | None
    on_sale: bool
    sale_price: float | None
    discount_percent: int | None
    is_featured: bool


def book_view(book: dict[str, Any]) -> BookView:
    def _to_int(x: Any) -> int | None:
        try:
            return int(x)
        except Exception:
            return None

    def _to_float(x: Any) -> float | None:
        try:
            return float(x)
        except Exception:
            return None

    return BookView(
        id=_to_int(book.get("id")),
        title=str(book.get("title", "")).strip(),
        author=str(book.get("author", "")).strip(),
        genre=str(book.get("genre", "")).strip(),
        rating=_to_float(book.get("rating")),
        pages=_to_int(book.get("pages")),
        price=_to_float(book.get("price")),
        year=_to_int(book.get("year")),
        description=str(book.get("description", "")).strip(),
        review_count=_to_int(book.get("reviewCount")),
        on_sale=bool(book.get("onSale", False)),
        sale_price=_to_float(book.get("salePrice")),
        discount_percent=_to_int(book.get("discountPercent")),
        is_featured=bool(book.get("isFeatured", False)),
    )


def effective_price(b: BookView) -> float | None:
    if b.on_sale and b.sale_price is not None:
        return b.sale_price
    return b.price


"""
RecommendBooks Tool

Purpose:
- Take in user preferences to recommend books to try next.

Note:
- Uses simple intent parsing + scoring against `storedata.json`.

How to think about this tool
----------------------------
This is intentionally **not** an embedding search or vector DB.
Instead it demonstrates a very common early-stage pattern:
- parse preferences from a natural-language request
- score each candidate in your structured dataset
- return a small ranked list with human-readable reasons
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain.tools import tool

from tools.storedata_utils import BookView, book_view, effective_price, fmt_money, load_store_books, norm


_NUM_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "book",
    "books",
    "by",
    "can",
    "could",
    "different",
    "do",
    "for",
    "from",
    "give",
    "i",
    "id",
    "i'd",
    "im",
    "in",
    "is",
    "it",
    "like",
    "me",
    "my",
    "next",
    "of",
    "on",
    "please",
    "recommend",
    "recommendations",
    "suggest",
    "suggestion",
    "suggestions",
    "that",
    "the",
    "to",
    "try",
    "want",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class _Prefs:
    # Parsed user preferences (all optional except count).
    # Keeping them in one dataclass makes it easy to pass around and test.
    count: int
    genres: tuple[str, ...]
    authors: tuple[str, ...]
    keywords: tuple[str, ...]
    on_sale: bool | None
    popularity: str | None  # "low" | "high" | None
    min_price: float | None
    max_price: float | None
    min_pages: int | None
    max_pages: int | None
    min_rating: float | None
    max_rating: float | None
    min_year: int | None
    max_year: int | None


def _unique_genres() -> list[str]:
    # Harvest genres from the inventory so we can detect them in user requests.
    seen: set[str] = set()
    out: list[str] = []
    for b in load_store_books():
        g = str(b.get("genre", "")).strip()
        if not g:
            continue
        gn = norm(g)
        if gn in seen:
            continue
        seen.add(gn)
        out.append(g)
    return out


def _extract_count(user_request: str) -> int:
    # How many recs did the user ask for? Default to 2.
    q = user_request.lower()
    # Prefer explicit "N books" / "N suggestions" / "top N" (avoid decimals like 4.7 stars)
    m = re.search(
        r"\b(?:top\s*)?(\d{1,2})\s*(?:books?|titles?|suggestions?|recs?|recommendations?|options?)\b",
        q,
    )
    if m:
        return max(1, min(10, int(m.group(1))))
    for w, n in _NUM_WORDS.items():
        if re.search(
            rf"\b{re.escape(w)}\s*(?:books?|titles?|suggestions?|recs?|recommendations?|options?)\b",
            q,
        ):
            return max(1, min(10, n))
    return 2


def _extract_price_bounds(user_request: str) -> tuple[float | None, float | None]:
    # Lightweight parsing of money cues. We only treat bare numbers as money
    # when the user also used a currency cue (e.g. "$", "dollars").
    q = user_request.lower()
    has_money_cue = ("$" in q) or any(k in q for k in ["dollar", "dollars", "usd", "bucks", "£", "€"])
    # between $x and $y
    m = re.search(r"\bbetween\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|-)\s*\$?\s*(\d+(?:\.\d+)?)\b", q)
    if m and has_money_cue:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    # $x-$y
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*-\s*\$\s*(\d+(?:\.\d+)?)", q)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    # under / below / less than
    m = re.search(r"\b(?:under|below|less than|at most|max(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)\b", q)
    if m and has_money_cue:
        return (None, float(m.group(1)))
    # over / at least / more than
    m = re.search(r"\b(?:over|above|more than|at least|min(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)\b", q)
    if m and has_money_cue:
        return (float(m.group(1)), None)
    return (None, None)


def _extract_page_bounds(user_request: str) -> tuple[int | None, int | None]:
    q = user_request.lower()
    m = re.search(r"\b(?:under|below|less than|at most|max(?:imum)?)\s*(\d{2,4})\s*pages?\b", q)
    if m:
        return (None, int(m.group(1)))
    m = re.search(r"\b(?:over|above|more than|at least|min(?:imum)?)\s*(\d{2,4})\s*pages?\b", q)
    if m:
        return (int(m.group(1)), None)
    # Soft cues
    if any(k in q for k in ["short", "quick", "fast read", "light read"]):
        return (None, 260)
    if any(k in q for k in ["long", "epic", "doorstopper"]):
        return (450, None)
    return (None, None)


def _extract_rating_bounds(user_request: str) -> tuple[float | None, float | None]:
    q = user_request.lower()
    has_rating_cue = any(k in q for k in ["star", "stars", "rating", "rated"])
    # "at least 4.5" / "4.5+"
    m = re.search(r"\b(?:at least|minimum|min|>=)\s*(\d(?:\.\d)?)\b", q)
    if m and has_rating_cue:
        try:
            return (float(m.group(1)), None)
        except Exception:
            pass
    m = re.search(r"\b(\d(?:\.\d)?)\s*\+\b", q)
    if m and has_rating_cue:
        try:
            return (float(m.group(1)), None)
        except Exception:
            pass
    # "under 4.2 stars" (rare, but support it)
    m = re.search(r"\b(?:under|below|less than|at most|max(?:imum)?|<=)\s*(\d(?:\.\d)?)\b", q)
    if m and has_rating_cue:
        try:
            return (None, float(m.group(1)))
        except Exception:
            pass
    # "rated 4.7" (treat as min 4.7 unless user says "around")
    m = re.search(r"\brated\s*(\d(?:\.\d)?)\b", q)
    if m:
        try:
            return (float(m.group(1)), None)
        except Exception:
            pass
    return (None, None)


def _extract_year_bounds(user_request: str) -> tuple[int | None, int | None]:
    q = user_request.lower()
    # Explicit: after/since 2015, before 1990, between 2010 and 2020
    m = re.search(r"\bbetween\s*(19\d{2}|20\d{2})\s*(?:and|-)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = re.search(r"\b(?:after|since|newer than|post)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        return (int(m.group(1)), None)
    m = re.search(r"\b(?:before|older than|pre)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        return (None, int(m.group(1)))
    m = re.search(r"\b(?:in|from)\s*(19\d{2}|20\d{2})\b", q)
    if m and any(k in q for k in ["published", "released", "publication", "came out", "year"]):
        y = int(m.group(1))
        return (y, y)
    # Soft cues
    if any(k in q for k in ["recent", "new", "newer", "modern"]):
        return (2018, None)
    if any(k in q for k in ["classic", "older", "old school"]):
        return (None, 2000)
    return (None, None)


def _extract_on_sale(user_request: str) -> bool | None:
    q = user_request.lower()
    if any(k in q for k in ["not on sale", "no sale", "not discounted"]):
        return False
    if any(k in q for k in ["on sale", "discount", "deal", "discounted"]):
        return True
    return None


def _extract_popularity(user_request: str) -> str | None:
    q = user_request.lower()
    if any(k in q for k in ["less known", "less-known", "under the radar", "hidden gem", "underrated", "not popular", "obscure"]):
        return "low"
    if any(k in q for k in ["popular", "bestseller", "best seller", "most reviewed", "trending"]):
        return "high"
    return None


def _extract_authors(user_request: str, books: list[BookView]) -> tuple[str, ...]:
    qn = norm(user_request)
    authors: list[str] = []
    seen: set[str] = set()
    for b in books:
        if not b.author:
            continue
        an = norm(b.author)
        if an and an in qn and an not in seen:
            seen.add(an)
            authors.append(b.author)
    return tuple(authors)


def _extract_genres_and_keywords(user_request: str, known_genres: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Step 1: detect explicit genres (from inventory) mentioned in the request.
    # Step 2: extract a few "keywords" to match against title/author/description.
    qn = norm(user_request)

    # Genre synonyms / common phrases
    qn = qn.replace("sci fi", "science fiction").replace("scifi", "science fiction").replace("science-fiction", "science fiction")
    qn = qn.replace("post apocalyptic", "post apocalyptic").replace("post-apocalyptic", "post apocalyptic")

    genres: list[str] = []
    seen_g: set[str] = set()
    for g in known_genres:
        gn = norm(g)
        if gn and gn in qn and gn not in seen_g:
            seen_g.add(gn)
            genres.append(gn)

    # Remove overly-broad genres that are substrings of more specific ones mentioned.
    # Example: if query contains "science fiction", we don't want to also treat "fiction" as a separate preference.
    genres_sorted = sorted(genres, key=len, reverse=True)
    cleaned: list[str] = []
    for g in genres_sorted:
        if any(g != kept and g in kept for kept in cleaned):
            continue
        cleaned.append(g)
    # Keep a stable, readable order (longest first is fine for coverage selection).
    genres = cleaned

    # Extract some keywords (kept intentionally lightweight)
    tokens = [t for t in qn.split() if t and t not in _STOPWORDS and not t.isdigit()]
    # Keep a few meaningful tokens
    kws: list[str] = []
    seen_k: set[str] = set()
    for t in tokens:
        if len(t) < 4:
            continue
        if t in seen_k:
            continue
        seen_k.add(t)
        kws.append(t)
        if len(kws) >= 8:
            break

    return (tuple(genres), tuple(kws))


def _review_percentiles(books: list[BookView]) -> dict[str, float]:
    """
    Return {normalized_title: percentile_in_[0,1]} based on review_count.
    Missing review_count -> treated as 0 reviews.
    """
    counts: list[int] = []
    title_to_count: dict[str, int] = {}
    for b in books:
        t = norm(b.title)
        if not t:
            continue
        c = int(b.review_count) if isinstance(b.review_count, int) else 0
        title_to_count[t] = c
        counts.append(c)

    if not counts:
        return {}

    sorted_counts = sorted(counts)
    n = len(sorted_counts)

    def percentile(count: int) -> float:
        if n <= 1:
            return 0.5
        # rank via first index (stable + fast for our small dataset)
        idx = 0
        for i, v in enumerate(sorted_counts):
            if v >= count:
                idx = i
                break
            idx = i
        return idx / float(n - 1)

    return {t: percentile(c) for t, c in title_to_count.items()}


def _score_book(b: BookView, prefs: _Prefs, *, review_percentile: float | None) -> tuple[float, list[str]]:
    """
    Score one book against preferences.

    Returns:
    - a numeric score (higher is better)
    - short reason fragments (used to justify recommendations)

    This is deliberately heuristic (weights are hand-tuned) so it's easy to understand.
    """
    score = 0.0
    reasons: list[str] = []

    text = norm(f"{b.title} {b.author} {b.genre} {b.description}")
    genre_n = norm(b.genre)

    # Author match
    if prefs.authors:
        if any(norm(a) == norm(b.author) for a in prefs.authors):
            score += 6.0
            reasons.append("matches your author")
        else:
            score -= 1.5

    # Genre match
    if prefs.genres:
        if any(g in genre_n for g in prefs.genres):
            score += 6.0
            reasons.append("fits your genre")
        elif any(g in text for g in prefs.genres):
            score += 3.0
            reasons.append("fits the vibe you're after")
        else:
            score -= 2.0

    # Keyword matches
    kw_hits = 0
    for kw in prefs.keywords:
        if kw and kw in text:
            kw_hits += 1
    if kw_hits:
        score += min(4.0, float(kw_hits))  # cap influence
        reasons.append("lines up with your keywords")

    # Sale preference
    if prefs.on_sale is True:
        if b.on_sale:
            score += 3.5
            reasons.append("on sale")
        else:
            score -= 2.5
    elif prefs.on_sale is False:
        score += 0.5 if not b.on_sale else -0.5

    # Price constraints (use effective price if present)
    p = effective_price(b)
    if prefs.max_price is not None and p is not None:
        score += 2.0 if p <= prefs.max_price else -2.5
    if prefs.min_price is not None and p is not None:
        score += 1.0 if p >= prefs.min_price else -1.5

    # Page constraints
    if prefs.max_pages is not None and b.pages is not None:
        score += 1.5 if b.pages <= prefs.max_pages else -1.5
    if prefs.min_pages is not None and b.pages is not None:
        score += 1.0 if b.pages >= prefs.min_pages else -1.0

    # Rating constraints
    if prefs.min_rating is not None and b.rating is not None:
        if b.rating >= prefs.min_rating:
            score += 2.0
            reasons.append("meets your rating bar")
        else:
            score -= 2.5
    if prefs.max_rating is not None and b.rating is not None:
        score += 1.0 if b.rating <= prefs.max_rating else -1.5

    # Year constraints
    if prefs.min_year is not None and b.year is not None:
        score += 1.5 if b.year >= prefs.min_year else -1.5
        if b.year >= prefs.min_year:
            reasons.append("fits your publish window")
    if prefs.max_year is not None and b.year is not None:
        score += 1.0 if b.year <= prefs.max_year else -1.5
        if b.year <= prefs.max_year and prefs.min_year is None:
            reasons.append("fits your publish window")

    # Popularity preference (review_count percentile as proxy)
    if prefs.popularity == "low" and review_percentile is not None:
        # Strongly prefer lower-percentile books.
        score += 3.0 * (1.0 - review_percentile)
        if review_percentile <= 0.35:
            reasons.append("less well-known")
    if prefs.popularity == "high" and review_percentile is not None:
        score += 3.0 * review_percentile
        if review_percentile >= 0.65:
            reasons.append("popular")

    # Light nudge for highly-rated books when otherwise similar
    if b.rating is not None:
        score += (b.rating - 4.2) * 0.6

    return score, reasons


def _opening_commentary(prefs: _Prefs, user_request: str) -> str:
    bits: list[str] = []
    if prefs.genres:
        # Show user we heard the genre(s) without being too literal.
        bits.append("genre vibes")
    if prefs.on_sale is True:
        bits.append("on-sale picks")
    if prefs.popularity == "low":
        bits.append("less-trodden titles")
    if prefs.max_price is not None:
        bits.append(f"under {fmt_money(prefs.max_price) or f'${prefs.max_price:.2f}'}")
    if prefs.max_pages is not None:
        bits.append(f"under {prefs.max_pages} pages")

    if "post apocalyptic" in norm(user_request) and "science fiction" in norm(user_request):
        return "Ooh. You like the dark and dreary - excellent choice."
    if prefs.popularity == "low":
        return "Splendid - I do love it when someone treads the less-beaten path."
    if prefs.on_sale is True:
        return "Marvellous - let's keep it on sale and well-matched to your taste."
    if bits:
        return "Got it - I'll aim for " + ", ".join(bits) + "."
    return "Got it - I'll pick a couple that should suit what you're after."


def _format_reco(b: BookView, *, prefs: _Prefs, user_request: str) -> str:
    # Format one recommendation as a short, skimmable paragraph.
    # 1 short descriptive sentence (avoid overly long descriptions)
    desc = b.description.strip()
    if desc and len(desc) > 160:
        desc = desc[:157].rstrip() + "..."

    price = fmt_money(effective_price(b))
    parts: list[str] = [f"{b.title} by {b.author}."]

    # If the user supplied criteria, echo the matched attributes to convey confidence.
    why_bits: list[str] = []
    if prefs.genres and b.genre:
        why_bits.append(b.genre)
    # If user asked for "post apocalyptic" but it's not the official genre, surface it when present in description.
    if "post apocalyptic" in norm(user_request) and "post apocalyptic" in norm(b.description):
        if not any("post" in x and "apocalyptic" in x for x in why_bits):
            why_bits.append("post-apocalyptic")
    if prefs.min_pages is not None or prefs.max_pages is not None:
        if b.pages is not None:
            why_bits.append(f"{b.pages} pages")
    if prefs.min_rating is not None or prefs.max_rating is not None:
        if b.rating is not None:
            why_bits.append(f"rated {b.rating:.1f}/5")
    if prefs.min_year is not None or prefs.max_year is not None:
        if b.year is not None:
            why_bits.append(f"published {b.year}")
    if prefs.on_sale is True:
        if b.on_sale and b.sale_price is not None:
            sp = fmt_money(b.sale_price)
            if sp:
                why_bits.append(f"on sale ({sp})")
        elif b.on_sale:
            why_bits.append("on sale")
    if prefs.max_price is not None:
        if price:
            why_bits.append(f"{price}")

    # Only add the "Why" line if the user actually mentioned constraints beyond "recommend/suggest".
    if any(
        [
            prefs.genres,
            prefs.on_sale is not None,
            prefs.popularity is not None,
            prefs.min_price is not None,
            prefs.max_price is not None,
            prefs.min_pages is not None,
            prefs.max_pages is not None,
            prefs.min_rating is not None,
            prefs.max_rating is not None,
            prefs.min_year is not None,
            prefs.max_year is not None,
        ]
    ) and why_bits:
        parts.append("Why: " + ", ".join(why_bits) + ".")

    if desc:
        parts.append(desc.rstrip(".") + ".")
    if b.on_sale and b.sale_price is not None:
        sale = fmt_money(b.sale_price) or ""
        if b.discount_percent:
            parts.append(f"Currently on sale for {sale} ({b.discount_percent}% off).".strip())
        else:
            parts.append(f"Currently on sale for {sale}.".strip())
    elif price:
        parts.append(f"Price: {price}.")
    return " ".join(p for p in parts if p).strip()


@tool
def recommend_books(user_request: str) -> str:
    """
    Recommend books based on user preferences.

    Input contract
    --------------
    - `user_request`: the user's natural language request (genres, constraints, vibes).

    Output contract
    ---------------
    - A multi-line string with an opener + N formatted recommendations.

    Calling convention
    ------------------
    This is a LangChain tool, so in Python it is invoked like:
    - `recommend_books.invoke({"user_request": "..."})`
    """
    books = [book_view(b) for b in load_store_books()]
    if not books:
        return "I can't load the inventory right now, I'm afraid."

    known_genres = _unique_genres()

    count = _extract_count(user_request)
    on_sale = _extract_on_sale(user_request)
    popularity = _extract_popularity(user_request)
    min_price, max_price = _extract_price_bounds(user_request)
    min_pages, max_pages = _extract_page_bounds(user_request)
    min_rating, max_rating = _extract_rating_bounds(user_request)
    min_year, max_year = _extract_year_bounds(user_request)

    genres, keywords = _extract_genres_and_keywords(user_request, known_genres)
    authors = _extract_authors(user_request, books)

    prefs = _Prefs(
        count=count,
        genres=genres,
        authors=authors,
        keywords=keywords,
        on_sale=on_sale,
        popularity=popularity,
        min_price=min_price,
        max_price=max_price,
        min_pages=min_pages,
        max_pages=max_pages,
        min_rating=min_rating,
        max_rating=max_rating,
        min_year=min_year,
        max_year=max_year,
    )

    percentiles = _review_percentiles(books)

    def _genre_match(b: BookView, pref_genre: str) -> bool:
        g = norm(b.genre)
        if pref_genre in g:
            return True
        txt = norm(f"{b.title} {b.description}")
        return pref_genre in txt

    def _meets_constraints_strict(b: BookView) -> bool:
        # "Strict" means: only include books that satisfy every explicit constraint the user stated.
        # If this yields too few candidates, we fall back to a softer approach (see below).
        if prefs.on_sale is True and not b.on_sale:
            return False
        if prefs.max_price is not None:
            p = effective_price(b)
            if p is None or p > prefs.max_price:
                return False
        if prefs.min_price is not None:
            p = effective_price(b)
            if p is None or p < prefs.min_price:
                return False
        if prefs.max_pages is not None:
            if b.pages is None or b.pages > prefs.max_pages:
                return False
        if prefs.min_pages is not None:
            if b.pages is None or b.pages < prefs.min_pages:
                return False
        if prefs.min_rating is not None:
            if b.rating is None or b.rating < prefs.min_rating:
                return False
        if prefs.max_rating is not None:
            if b.rating is None or b.rating > prefs.max_rating:
                return False
        if prefs.min_year is not None:
            if b.year is None or b.year < prefs.min_year:
                return False
        if prefs.max_year is not None:
            if b.year is None or b.year > prefs.max_year:
                return False
        if prefs.genres:
            if not any(_genre_match(b, g) for g in prefs.genres):
                return False
        if prefs.authors:
            if not any(norm(a) == norm(b.author) for a in prefs.authors):
                return False
        return True

    strict_matches = [b for b in books if _meets_constraints_strict(b)]
    used_fallback = len(strict_matches) < prefs.count

    scored: list[tuple[float, BookView]] = []
    def _soft_filter(cands: list[BookView], filtered: list[BookView]) -> list[BookView]:
        return filtered if len(filtered) >= prefs.count else cands

    if len(strict_matches) >= prefs.count:
        candidate_books = strict_matches
    else:
        # Fallback strategy: if strict matches are insufficient, anchor on genre when provided.
        if prefs.genres:
            genre_anchor = [b for b in books if any(_genre_match(b, g) for g in prefs.genres)]
            candidate_books = genre_anchor if genre_anchor else books
        else:
            candidate_books = books
    # Apply "hard when possible" filters: try to honor explicit constraints without producing 0 results.
    if prefs.on_sale is True:
        candidate_books = _soft_filter(candidate_books, [b for b in candidate_books if b.on_sale])
    if prefs.max_price is not None:
        candidate_books = _soft_filter(
            candidate_books,
            [b for b in candidate_books if (effective_price(b) is not None and effective_price(b) <= prefs.max_price)],
        )
    if prefs.max_pages is not None:
        candidate_books = _soft_filter(
            candidate_books, [b for b in candidate_books if (b.pages is not None and b.pages <= prefs.max_pages)]
        )
    if prefs.min_rating is not None:
        candidate_books = _soft_filter(
            candidate_books, [b for b in candidate_books if (b.rating is not None and b.rating >= prefs.min_rating)]
        )
    if prefs.min_year is not None:
        candidate_books = _soft_filter(
            candidate_books, [b for b in candidate_books if (b.year is not None and b.year >= prefs.min_year)]
        )
    if prefs.max_year is not None:
        candidate_books = _soft_filter(
            candidate_books, [b for b in candidate_books if (b.year is not None and b.year <= prefs.max_year)]
        )

    if prefs.genres:
        genre_only = [b for b in candidate_books if any(_genre_match(b, g) for g in prefs.genres)]
        candidate_books = _soft_filter(candidate_books, genre_only)

    for b in candidate_books:
        s, _ = _score_book(b, prefs, review_percentile=percentiles.get(norm(b.title)))
        scored.append((s, b))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top-N, keeping titles unique; try to avoid repeating author if user asked "different".
    want_different = "different" in user_request.lower()
    picked: list[BookView] = []
    seen_titles: set[str] = set()
    seen_authors: set[str] = set()

    # If the user mentioned multiple genres, try to cover them across picks (especially when asking for 2).
    if prefs.genres and len(prefs.genres) >= 2 and prefs.count >= 2:
        for g in prefs.genres:
            for _, b in scored:
                t = norm(b.title)
                if not t or t in seen_titles:
                    continue
                if not _genre_match(b, g):
                    continue
                if want_different and b.author and norm(b.author) in seen_authors and len(picked) < max(2, prefs.count):
                    continue
                picked.append(b)
                seen_titles.add(t)
                if b.author:
                    seen_authors.add(norm(b.author))
                if len(picked) >= prefs.count:
                    break
            if len(picked) >= prefs.count:
                break

    # Fill remaining slots by overall score
    for _, b in scored:
        if len(picked) >= prefs.count:
            break
        t = norm(b.title)
        if not t or t in seen_titles:
            continue
        if want_different and b.author and norm(b.author) in seen_authors and len(picked) < max(2, prefs.count):
            continue
        picked.append(b)
        seen_titles.add(t)
        if b.author:
            seen_authors.add(norm(b.author))

    if not picked:
        return "I couldn't find any good matches in our current inventory based on that request."

    opener = _opening_commentary(prefs, user_request)
    if used_fallback and any(
        [
            prefs.genres,
            prefs.on_sale is not None,
            prefs.popularity is not None,
            prefs.min_price is not None,
            prefs.max_price is not None,
            prefs.min_pages is not None,
            prefs.max_pages is not None,
            prefs.min_rating is not None,
            prefs.max_rating is not None,
            prefs.min_year is not None,
            prefs.max_year is not None,
        ]
    ):
        opener = (
            opener
            + " We don't have many exact matches for every constraint, so these are the closest fits from what we do have."
        )
    header = "Here are a couple you might enjoy:" if prefs.count == 2 else f"Here are {prefs.count} you might enjoy:"

    lines: list[str] = [opener, "", header, ""]
    for b in picked:
        lines.append(_format_reco(b, prefs=prefs, user_request=user_request))
        lines.append("")

    return "\n".join(lines).rstrip()


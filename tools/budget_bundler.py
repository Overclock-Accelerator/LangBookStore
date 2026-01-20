"""
BudgetBundler Tool

Purpose:
- Take in a budget and interests and suggest a bundle of books to purchase.

Note:
- Uses a knapsack-style dynamic programming solver (in cents) to maximize bundle size under budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain.tools import tool

from tools.storedata_utils import BookView, book_view, effective_price, fmt_money, load_store_books, norm


@dataclass(frozen=True)
class _BundlePrefs:
    budget: float
    genres: tuple[str, ...]
    require_genre_mix: bool
    recent: bool
    min_year: int | None
    max_year: int | None
    on_sale: bool | None
    min_rating: float | None
    max_pages: int | None
    popularity: str | None  # "low" | "high" | None


def _current_year() -> int:
    # Keep deterministic & simple; inventory is static and the runtime date isn't critical here.
    return 2026


def _extract_budget(user_request: str) -> float | None:
    q = user_request.lower()
    # Prefer $ amounts
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)", q)
    if m:
        return float(m.group(1))
    # "budget of 65"
    m = re.search(r"\bbudget(?:\s+of)?\s*(\d+(?:\.\d+)?)\b", q)
    if m:
        return float(m.group(1))
    # "under 65 dollars"
    m = re.search(r"\b(?:under|within|less than|at most|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd)\b", q)
    if m:
        return float(m.group(1))
    return None


def _unique_genres() -> list[str]:
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


def _extract_genres(user_request: str, known_genres: list[str]) -> tuple[str, ...]:
    qn = norm(user_request)
    qn = qn.replace("sci fi", "science fiction").replace("scifi", "science fiction").replace("science-fiction", "science fiction")
    qn = qn.replace("post-apocalyptic", "post apocalyptic")

    genres: list[str] = []
    seen: set[str] = set()
    for g in known_genres:
        gn = norm(g)
        if gn and gn in qn and gn not in seen:
            seen.add(gn)
            genres.append(gn)

    # Remove overly-broad genres that are substrings of more specific ones mentioned.
    genres_sorted = sorted(genres, key=len, reverse=True)
    cleaned: list[str] = []
    for g in genres_sorted:
        if any(g != kept and g in kept for kept in cleaned):
            continue
        cleaned.append(g)
    return tuple(cleaned)


def _extract_recent_and_years(user_request: str) -> tuple[bool, int | None, int | None]:
    q = user_request.lower()
    cy = _current_year()
    if "recent" in q or "new" in q or "newer" in q or "modern" in q:
        # Interpreted as "within last ~10 years", matching the sample where 2017 counts as recent.
        return True, cy - 10, None

    m = re.search(r"\b(?:after|since)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        return False, int(m.group(1)), None
    m = re.search(r"\b(?:before)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        return False, None, int(m.group(1))
    m = re.search(r"\bbetween\s*(19\d{2}|20\d{2})\s*(?:and|-)\s*(19\d{2}|20\d{2})\b", q)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return False, min(lo, hi), max(lo, hi)
    return False, None, None


def _extract_on_sale(user_request: str) -> bool | None:
    q = user_request.lower()
    if any(k in q for k in ["not on sale", "no sale", "not discounted"]):
        return False
    # Include common variants like "on-sale" / "onsale".
    if any(k in q for k in ["on sale", "on-sale", "onsale", "sale", "discount", "deal", "discounted"]):
        return True
    return None


def _extract_min_rating(user_request: str) -> float | None:
    q = user_request.lower()
    if not any(k in q for k in ["star", "stars", "rating", "rated"]):
        return None
    m = re.search(r"\b(?:at least|minimum|min|>=)\s*(\d(?:\.\d)?)\b", q)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d(?:\.\d)?)\s*\+\b", q)
    if m:
        return float(m.group(1))
    m = re.search(r"\brated\s*(\d(?:\.\d)?)\b", q)
    if m:
        return float(m.group(1))
    return None


def _extract_max_pages(user_request: str) -> int | None:
    q = user_request.lower()
    m = re.search(r"\b(?:under|below|less than|at most|max(?:imum)?)\s*(\d{2,4})\s*pages?\b", q)
    if m:
        return int(m.group(1))
    if any(k in q for k in ["short", "quick", "fast read", "light read"]):
        return 260
    return None


def _extract_popularity(user_request: str) -> str | None:
    q = user_request.lower()
    if any(k in q for k in ["less known", "less-known", "under the radar", "hidden gem", "underrated", "not popular", "obscure"]):
        return "low"
    if any(k in q for k in ["popular", "bestseller", "best seller", "most reviewed", "trending"]):
        return "high"
    return None


def _wants_mix(user_request: str) -> bool:
    q = user_request.lower()
    return any(k in q for k in ["mix of", "mix", "variety", "one of each", "a range of"])


def _price_cents(b: BookView) -> int | None:
    p = effective_price(b)
    if p is None:
        return None
    return int(round(float(p) * 100))


def _median(nums: list[int]) -> int:
    if not nums:
        return 0
    s = sorted(nums)
    return s[len(s) // 2]


def _filter_candidates(books: list[BookView], prefs: _BundlePrefs) -> list[BookView]:
    out: list[BookView] = []
    review_counts = [b.review_count for b in books if isinstance(b.review_count, int)]
    median_reviews = _median([int(x) for x in review_counts if x is not None])
    for b in books:
        if prefs.genres and norm(b.genre) not in prefs.genres:
            continue
        if prefs.on_sale is True and not b.on_sale:
            continue
        if prefs.on_sale is False and b.on_sale:
            continue
        if prefs.min_year is not None:
            if b.year is None or b.year < prefs.min_year:
                continue
        if prefs.max_year is not None:
            if b.year is None or b.year > prefs.max_year:
                continue
        if prefs.min_rating is not None:
            if b.rating is None or b.rating < prefs.min_rating:
                continue
        if prefs.max_pages is not None:
            if b.pages is None or b.pages > prefs.max_pages:
                continue
        if prefs.popularity == "low" and b.review_count is not None and median_reviews:
            if b.review_count >= median_reviews:
                continue
        if prefs.popularity == "high" and b.review_count is not None and median_reviews:
            if b.review_count < median_reviews:
                continue
        if _price_cents(b) is None:
            continue
        out.append(b)
    return out


def _solve_bundle_max_books(
    candidates: list[BookView],
    *,
    budget_cents: int,
    required_genres: tuple[str, ...],
) -> list[BookView]:
    """
    0/1 knapsack with optional required-genre coverage.
    Objective: maximize (book_count, total_spend) under budget.
    """
    # Map required genres to mask bits
    genre_to_bit = {g: (1 << i) for i, g in enumerate(required_genres)}
    full_mask = (1 << len(required_genres)) - 1

    items: list[tuple[int, int, BookView]] = []  # (price_cents, mask, book)
    for b in candidates:
        pc = _price_cents(b)
        if pc is None or pc <= 0 or pc > budget_cents:
            continue
        mask = 0
        if required_genres:
            g = norm(b.genre)
            if g in genre_to_bit:
                mask |= genre_to_bit[g]
        items.append((pc, mask, b))

    # dp[mask][c] = (count, spend, prev_mask, prev_c, item_index) or None
    dp: list[list[tuple[int, int, int, int, int] | None]] = [
        [None] * (budget_cents + 1) for _ in range(max(1, 1 << len(required_genres)))
    ]
    dp[0][0] = (0, 0, -1, -1, -1)

    def better(a: tuple[int, int, int, int, int], b: tuple[int, int, int, int, int]) -> bool:
        # True if a is better than b
        if a[0] != b[0]:
            return a[0] > b[0]
        return a[1] > b[1]

    for idx, (pc, m, _) in enumerate(items):
        for mask in range(len(dp) - 1, -1, -1):
            for c in range(budget_cents - pc, -1, -1):
                cur = dp[mask][c]
                if cur is None:
                    continue
                new_mask = mask | m
                new_c = c + pc
                cand = (cur[0] + 1, new_c, mask, c, idx)
                existing = dp[new_mask][new_c]
                if existing is None or better(cand, existing):
                    dp[new_mask][new_c] = cand

    # Pick best end state
    target_masks = [full_mask] if full_mask != 0 else [0]
    best_state: tuple[int, int, int, int, int] | None = None
    best_mask = 0
    for tm in target_masks:
        for c in range(budget_cents + 1):
            st = dp[tm][c]
            if st is None:
                continue
            if best_state is None or better(st, best_state):
                best_state = st
                best_mask = tm

    # If we couldn't satisfy full coverage, fall back to best across all masks.
    if best_state is None and full_mask != 0:
        for tm in range(len(dp)):
            for c in range(budget_cents + 1):
                st = dp[tm][c]
                if st is None:
                    continue
                if best_state is None or better(st, best_state):
                    best_state = st
                    best_mask = tm

    if best_state is None:
        return []

    # Reconstruct
    picked: list[BookView] = []
    st = best_state
    mask = best_mask
    c = st[1]
    while st is not None and st[4] != -1:
        _, _, prev_mask, prev_c, item_idx = st
        _, _, b = items[item_idx]
        picked.append(b)
        mask, c = prev_mask, prev_c
        st = dp[mask][c]
    picked.reverse()
    return picked


@tool
def budget_bundler(budget_request: str) -> str:
    """Assemble a suggested book order within a stated budget."""
    budget = _extract_budget(budget_request)
    if budget is None or budget <= 0:
        return "Please tell me your budget (e.g., '$65') and what kinds of books you'd like."

    books = [book_view(b) for b in load_store_books()]
    if not books:
        return "I can't load the inventory right now, I'm afraid."

    known_genres = _unique_genres()
    genres = _extract_genres(budget_request, known_genres)
    require_mix = _wants_mix(budget_request) and len(genres) >= 2
    recent, min_year, max_year = _extract_recent_and_years(budget_request)
    on_sale = _extract_on_sale(budget_request)
    min_rating = _extract_min_rating(budget_request)
    max_pages = _extract_max_pages(budget_request)
    popularity = _extract_popularity(budget_request)

    prefs = _BundlePrefs(
        budget=budget,
        genres=genres,
        require_genre_mix=require_mix,
        recent=recent,
        min_year=min_year,
        max_year=max_year,
        on_sale=on_sale,
        min_rating=min_rating,
        max_pages=max_pages,
        popularity=popularity,
    )

    candidates = _filter_candidates(books, prefs)
    if not candidates:
        return "I couldn't find any books that match those criteria within our current inventory."

    budget_cents = int(round(budget * 100))
    required_genres = prefs.genres if prefs.require_genre_mix else tuple()
    picked = _solve_bundle_max_books(candidates, budget_cents=budget_cents, required_genres=required_genres)

    if not picked:
        return "I couldn't build a bundle under that budget with the requested criteria."

    total_cents = sum(_price_cents(b) or 0 for b in picked)
    total_str = fmt_money(total_cents / 100.0) or f"${total_cents/100.0:.2f}"

    criteria_bits: list[str] = []
    if prefs.genres:
        criteria_bits.append(" / ".join([g.title() if g.islower() else g for g in prefs.genres]))
    if prefs.recent and prefs.min_year is not None:
        criteria_bits.append(f"recent (since {prefs.min_year})")
    if prefs.on_sale is True:
        criteria_bits.append("on sale")
    if prefs.popularity == "low":
        criteria_bits.append("less well-known")
    if prefs.min_rating is not None:
        criteria_bits.append(f"rated {prefs.min_rating:.1f}+")
    if prefs.max_pages is not None:
        criteria_bits.append(f"under {prefs.max_pages} pages")

    intro = (
        "Certainly. Here's a recommended bundle."
        if not criteria_bits
        else "Certainly. Here's a recommended bundle based on " + ", ".join(criteria_bits) + "."
    )
    lines: list[str] = [intro, ""]
    for b in picked:
        p = effective_price(b)
        p_str = fmt_money(p) if p is not None else "N/A"
        yr = str(b.year) if b.year is not None else "N/A"
        lines.append(f"{b.title} ({yr}): {p_str}")
    lines.append("")
    lines.append(f"Order Total: {total_str}")
    return "\n".join(lines).strip()


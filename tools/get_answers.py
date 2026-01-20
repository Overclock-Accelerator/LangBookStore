"""
GetAnswers Tool (placeholder)

Purpose:
- Answer questions about the bookstore’s current books (availability, prices, sales, etc.)

Note:
- Now implemented for title-based questions against `storedata.json`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain.tools import tool

from tools.storedata_utils import BookView as _SharedBookView
from tools.storedata_utils import book_view as _shared_book_view
from tools.storedata_utils import fmt_money as _shared_fmt_money
from tools.storedata_utils import load_store_books as _load_store_books
from tools.storedata_utils import norm as _norm_shared

def _load_storedata() -> list[dict[str, Any]]:
    return _load_store_books()


def _norm(s: str) -> str:
    return _norm_shared(s)


def _find_books_by_title(user_query: str) -> list[dict[str, Any]]:
    """
    Find the best matching book by detecting a title mention in the user's query.

    Strategy (simple + robust):
    - Normalize query and titles (lowercase, strip punctuation)
    - Choose the *longest* title that appears as a substring in the query
    """
    qn = _norm(user_query)
    best_len = 0
    matches: list[dict[str, Any]] = []
    for book in _load_storedata():
        title = str(book.get("title", "")).strip()
        if not title:
            continue
        tn = _norm(title)
        if not tn or tn not in qn:
            continue
        score = len(tn)
        if score > best_len:
            best_len = score
            matches = [book]
        elif score == best_len:
            matches.append(book)
    return matches


def _wants_any_details(q: str) -> bool:
    q = q.lower()
    return any(
        k in q
        for k in [
            "author",
            "who wrote",
            "genre",
            "rating",
            "stars",
            "pages",
            "page",
            "price",
            "cost",
            "how much",
            "year",
            "released",
            "published",
            "description",
            "about",
            "summary",
            "review",
            "on sale",
            "sale",
            "discount",
        ]
    )


def _extract_requested_fields(user_query: str) -> set[str]:
    """
    Extract which specific fields the user is asking about.

    Returns canonical field names:
    availability, description, author, genre, rating, pages, year, price, sale, discount, reviews
    """
    q = user_query.lower()
    fields: set[str] = set()

    if any(k in q for k in ["do you have", "have you got", "in stock", "available", "carry"]):
        fields.add("availability")

    if any(k in q for k in ["description", "about", "summary", "synopsis", "blurb", "what is it about"]):
        fields.add("description")

    if any(k in q for k in ["author", "who wrote", "written by"]):
        fields.add("author")

    if any(k in q for k in ["genre", "category", "what kind", "type of book"]):
        fields.add("genre")

    if any(k in q for k in ["rating", "rated", "stars"]):
        fields.add("rating")

    if any(
        k in q
        for k in [
            "pages",
            "page",
            "page count",
            "how many pages",
            "how long is",
            "length",
        ]
    ):
        fields.add("pages")

    if any(k in q for k in ["year", "released", "published", "publication", "when did", "came out", "when was"]):
        fields.add("year")

    if any(k in q for k in ["price", "cost", "how much", "how much is", "how much does"]):
        fields.add("price")

    # Sale / discount: keep these distinct so "discount percent" questions don't get treated
    # as generic "sale?" intent.
    if any(k in q for k in ["on sale", "sale price", "deal", "discounted"]):
        fields.add("sale")

    if any(k in q for k in ["discount", "discount percent", "% off", "percent off"]):
        fields.add("discount")

    if any(k in q for k in ["review", "reviews", "review count", "how many reviews"]):
        fields.add("reviews")

    return fields


def _fmt_money(value: Any) -> str | None:
    return _shared_fmt_money(value)


@dataclass(frozen=True)
class _BookView:
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


def _book_view(book: dict[str, Any]) -> _BookView:
    # Keep this tool's view stable, but source parsing from shared helper.
    sb: _SharedBookView = _shared_book_view(book)
    return _BookView(
        title=sb.title,
        author=sb.author,
        genre=sb.genre,
        rating=sb.rating,
        pages=sb.pages,
        price=sb.price,
        year=sb.year,
        description=sb.description,
        review_count=sb.review_count,
        on_sale=sb.on_sale,
        sale_price=sb.sale_price,
        discount_percent=sb.discount_percent,
    )


def _compose_response(user_query: str, b: _BookView) -> str:
    requested = _extract_requested_fields(user_query)

    asked_price = "price" in requested
    asked_author = "author" in requested
    asked_genre = "genre" in requested
    asked_rating = "rating" in requested
    asked_pages = "pages" in requested
    asked_year = "year" in requested
    asked_desc = "description" in requested
    asked_reviews = "reviews" in requested
    asked_sale = "sale" in requested
    asked_availability = "availability" in requested
    asked_discount = "discount" in requested

    asked_any_detail = _wants_any_details(user_query)

    # Special-case: if the user asked about discount percent, answer discount directly even if
    # the phrasing also implies a sale question.
    if asked_discount and not asked_price and requested.issubset({"discount", "sale"}):
        sale_price = _fmt_money(b.sale_price) if (b.on_sale and b.sale_price is not None) else None
        if b.on_sale and b.discount_percent:
            if sale_price:
                return f"{b.title} is {b.discount_percent}% off (sale price {sale_price})."
            return f"{b.title} is {b.discount_percent}% off."
        return f"I don't have a discount listed for {b.title}."

    # If the user asked for exactly one concrete attribute, answer it directly
    # (avoid leading with stock/description unless asked).
    if len(requested) == 1:
        only = next(iter(requested))
        if only == "pages":
            if b.pages is not None:
                return f"{b.title} is {b.pages} pages."
            return f"I don't have the page count on file for {b.title}."
        if only == "author":
            if b.author:
                return f"{b.title} is by {b.author}."
            return f"I don't have the author on file for {b.title}."
        if only == "genre":
            if b.genre:
                return f"{b.title} is {b.genre}."
            return f"I don't have the genre on file for {b.title}."
        if only == "rating":
            if b.rating is not None:
                return f"{b.title} is rated {b.rating:.1f} out of 5."
            return f"I don't have a rating on file for {b.title}."
        if only == "year":
            if b.year is not None:
                return f"{b.title} was released in {b.year}."
            return f"I don't have a release year on file for {b.title}."
        if only == "description":
            if b.description:
                return f"{b.title}: {b.description.rstrip('.') }."
            return f"I don't have a description on file for {b.title}."
        if only == "reviews":
            if b.review_count is not None:
                return f"{b.title} has {b.review_count:,} reviews."
            return f"I don't have the review count on file for {b.title}."
        if only == "availability":
            return f"Yes - we have {b.title} in stock."
        if only in {"price", "sale", "discount"}:
            base_price = _fmt_money(b.price)
            sale_price = _fmt_money(b.sale_price) if (b.on_sale and b.sale_price is not None) else None
            if only == "price":
                if sale_price:
                    if b.discount_percent:
                        return f"{b.title} is on sale for {sale_price} ({b.discount_percent}% off)."
                    return f"{b.title} is on sale for {sale_price}."
                if base_price:
                    return f"{b.title} costs {base_price}."
                return f"I don't have the price on file for {b.title}."
            if only == "sale":
                if sale_price:
                    if b.discount_percent:
                        return f"Yes - {b.title} is on sale for {sale_price} ({b.discount_percent}% off)."
                    return f"Yes - {b.title} is on sale for {sale_price}."
                return f"No - {b.title} isn't on sale at the moment."
            # only == "discount"
            if b.on_sale and b.discount_percent:
                if sale_price:
                    return f"{b.title} is {b.discount_percent}% off (sale price {sale_price})."
                return f"{b.title} is {b.discount_percent}% off."
            return f"I don't have a discount listed for {b.title}."

    def primary_intent() -> str:
        if asked_year and not (asked_price or asked_sale or asked_desc or asked_pages or asked_rating or asked_reviews):
            return "year"
        if asked_price:
            return "price"
        if asked_sale:
            return "sale"
        if asked_availability and not asked_any_detail:
            return "availability"
        return "general"

    lines: list[str] = []
    intent = primary_intent()

    # Lead with what the user asked for (more conversational, less boilerplate).
    if intent == "year":
        if b.year is not None:
            lines.append(f"{b.title} was released in {b.year}.")
        else:
            lines.append(f"I don't have a release year on file for {b.title}, I'm afraid.")
    elif intent in {"price", "sale", "availability"}:
        lines.append(f"Yes! We have {b.title} in stock.")
    else:
        lines.append(f"Certainly - {b.title} is in stock.")

    # Weave in a short line from the description when it's helpful.
    if (intent in {"availability", "price", "general"} or asked_desc) and b.description:
        lines.append(b.description.rstrip(".") + ".")

    # Only include genre/year when the question points that way.
    if asked_genre and b.genre:
        lines.append(f"Genre: {b.genre}.")
    if asked_year and intent != "year" and b.year is not None:
        lines.append(f"Released: {b.year}.")

    if asked_author and b.author:
        lines.append(f"It is by {b.author}.")

    if asked_price:
        current = _fmt_money(b.sale_price if (b.on_sale and b.sale_price is not None) else b.price)
        if current:
            if b.on_sale and b.sale_price is not None:
                if b.discount_percent:
                    lines.append(f"It's currently on sale for {current} ({b.discount_percent}% off).")
                else:
                    lines.append(f"It's currently on sale for {current}.")
            else:
                lines.append(f"It is currently selling for {current}.")

    if asked_sale and not asked_price:
        if b.on_sale and b.sale_price is not None:
            current = _fmt_money(b.sale_price)
            if current:
                lines.append(f"Yes - it's on sale at the moment for {current}.")
        else:
            lines.append("It isn't on sale at the moment.")

    if asked_discount and not (asked_price or asked_sale):
        if b.on_sale and b.discount_percent:
            if b.sale_price is not None:
                current = _fmt_money(b.sale_price)
                if current:
                    lines.append(f"Discount: {b.discount_percent}% off (sale price {current}).")
                else:
                    lines.append(f"Discount: {b.discount_percent}% off.")
            else:
                lines.append(f"Discount: {b.discount_percent}% off.")
        else:
            lines.append("I don't have a discount listed at the moment.")

    if asked_rating and b.rating is not None:
        lines.append(f"It's rated {b.rating:.1f} out of 5.")

    if asked_reviews and b.review_count is not None:
        lines.append(f"It has {b.review_count:,} reviews.")

    if asked_pages and b.pages is not None:
        lines.append(f"It's {b.pages} pages.")

    if asked_desc and b.description:
        # Already included above; don't repeat.
        pass

    return " ".join(lines).strip()


@tool
def get_answers(query: str) -> str:
    """Answer questions about the bookstore’s current books by title."""
    books = _find_books_by_title(query)
    if not books:
        return (
            "I'm afraid I can't find that title in our current inventory - "
            "it appears we don't have it in stock at the moment."
        )

    if len(books) > 1:
        options: list[str] = []
        for b in books[:5]:
            bv = _book_view(b)
            price = _fmt_money(bv.sale_price if (bv.on_sale and bv.sale_price is not None) else bv.price) or "N/A"
            yr = str(bv.year) if bv.year is not None else "N/A"
            options.append(f"- {bv.title} by {bv.author} ({bv.genre}, {yr}) - {price}")
        return (
            "I can see more than one matching entry for that title in our inventory. "
            "Which one did you mean?\n" + "\n".join(options)
        )

    return _compose_response(query, _book_view(books[0]))


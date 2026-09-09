"""
fashion_tool.py

Self-contained product search tool for the Dress & Fashion Finder agent,
backed by DuckDuckGo's free, key-less search endpoints via the `ddgs`
package (previously published as `duckduckgo_search`).

No API key of any kind is required to use this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

# The package was renamed from `duckduckgo_search` to `ddgs`; support both so
# the app runs against whichever version is already installed.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - older installs
    from duckduckgo_search import DDGS  # type: ignore[no-redef]

MAX_RESULTS_DEFAULT = 9
SEARCH_ATTEMPTS = 2
SEARCH_REGION = "in-en"

# Results hosted on real storefronts are far more useful than blog or
# Pinterest hits, so they are floated to the top of the grid.
RETAIL_DOMAINS = (
    "amazon.",
    "myntra.",
    "ajio.",
    "flipkart.",
    "nykaafashion.",
    "meesho.",
    "tatacliq.",
    "zara.",
    "hm.com",
    "shein.",
    "asos.",
    "nordstrom.",
    "macys.",
    "westside.",
    "fabindia.",
    "walmart.",
    "etsy.",
    "target.",
    "revolve.",
    "boohoo.",
    "namshi.",
    "uniqlo.",
)

# Social/CDN hosts that surface pretty pictures but never a buyable page.
BLOCKED_DOMAINS = (
    "pinterest.",
    "instagram.",
    "facebook.",
    "tumblr.",
    "lookaside.",
    "storage.googleapis.com",
)


class ProductSearchError(Exception):
    """Raised whenever a product search cannot be completed."""


@dataclass(frozen=True)
class Product:
    """A single shoppable search result."""

    title: str
    image_url: str
    product_url: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "image_url": self.image_url,
            "product_url": self.product_url,
            "source": self.source,
        }


def _domain_of(url: str) -> str:
    """Return the bare host for a URL, or an empty string if it is unparseable."""
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def _is_retail(url: str) -> bool:
    domain = _domain_of(url)
    return any(retailer in domain for retailer in RETAIL_DOMAINS)


def _is_blocked(url: str) -> bool:
    domain = _domain_of(url)
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def _rank(products: list[Product]) -> list[Product]:
    """Sort storefront results ahead of everything else, keeping relative order."""
    return sorted(products, key=lambda product: not _is_retail(product.product_url))


def build_shopping_query(keywords: str) -> str:
    """Turn refined keywords into a query biased toward buyable product pages."""
    cleaned = " ".join(keywords.split())
    if not cleaned:
        raise ProductSearchError("The search query was empty.")
    # "buy online price" pushes DuckDuckGo toward storefront product pages
    # rather than lookbooks, and excluding Pinterest removes most of the
    # image-only noise that has no purchase link behind it.
    return f"{cleaned} buy online price -pinterest"


def _relevant(product: "Product", keywords: str) -> bool:
    """
    True if the result plausibly matches the search.

    DuckDuckGo's image backend occasionally answers with a page of entirely
    unrelated products, so require at least one meaningful keyword token to
    appear in the title or the URL.
    """
    tokens = [token for token in keywords.lower().split() if len(token) >= 4]
    if not tokens:
        return True
    haystack = f"{product.title} {product.product_url}".lower()
    return any(token in haystack for token in tokens)


def _fetch_images(query: str, max_results: int, region: str) -> list[dict[str, Any]]:
    """Run one image search, translating any backend failure into our own error."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.images(query, region=region, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - network/parse failures vary by version
        raise ProductSearchError(f"Product search failed: {exc}") from exc


def _to_products(raw_results: list[dict[str, Any]]) -> list["Product"]:
    """Keep only shoppable, de-duplicated results and wrap them as Products."""
    products: list[Product] = []
    seen_images: set[str] = set()
    for result in raw_results:
        image_url = (result.get("image") or result.get("thumbnail") or "").strip()
        product_url = (result.get("url") or "").strip()
        if not image_url or not product_url or image_url in seen_images:
            continue
        if _is_blocked(product_url):
            continue
        seen_images.add(image_url)
        products.append(
            Product(
                title=(result.get("title") or "Untitled item").strip(),
                image_url=image_url,
                product_url=product_url,
                source=result.get("source") or _domain_of(product_url),
            )
        )
    return products


def search_products(
    keywords: str,
    max_results: int = MAX_RESULTS_DEFAULT,
    region: str = SEARCH_REGION,
) -> list[dict[str, str]]:
    """
    Search the web for shoppable fashion products matching `keywords`.

    Returns a list of dicts with `title`, `image_url`, `product_url` and
    `source`. Raises ProductSearchError if the search itself fails; an empty
    list means the search worked but matched nothing usable.
    """
    query = build_shopping_query(keywords)

    # One retry: an occasional DuckDuckGo response is entirely off-topic, and
    # asking again almost always returns the real product set.
    for _attempt in range(SEARCH_ATTEMPTS):
        raw_results = _fetch_images(query, max_results * 2, region)
        products = [
            product
            for product in _to_products(raw_results)
            if _relevant(product, keywords)
        ]
        if products:
            return [product.as_dict() for product in _rank(products)[:max_results]]

    return []


def search_products_safe(
    keywords: str,
    max_results: int = MAX_RESULTS_DEFAULT,
) -> tuple[list[dict[str, str]], Optional[str]]:
    """Wrapper for UI use: returns (products, error_message) instead of raising."""
    try:
        return search_products(keywords, max_results=max_results), None
    except ProductSearchError as exc:
        return [], str(exc)

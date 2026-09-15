"""te.eg crawler.

Verified against the live site before this was written:
  * te.eg is fully server-rendered — 5-11k visible chars per page — so plain
    HTTP + a parser is enough. No Playwright, which saves ~400 MB and an hour.
  * robots.txt is `User-Agent: * / Disallow:` — everything is permitted. We are
    still polite: one worker, a delay between requests, a real UA string.
  * Arabic and English are separate URL paths (`/ar/...` vs the default), so
    both language versions are reachable by ordinary link-following (B.3).
  * `/personal/sitemap/` exists and is a good breadth seed.

The crawl is a one-time snapshot committed to the repo, so Colab clones and runs
without re-crawling (see plan).
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

log = logging.getLogger(__name__)

USER_AGENT = (
    "TelecomEgyptAssistantBot/0.1 (POC knowledge-base build; contact: site owner) "
    "python-httpx"
)

ALLOWED_HOSTS = {"te.eg", "www.te.eg"}

# Liferay serves its asset pipeline under these prefixes. They are CSS/JS/icon
# bundles, never content, and they dominate the link graph if not excluded.
SKIP_PATH_PREFIXES = (
    "/o/", "/combo", "/documents/", "/c/portal/", "/image/", "/webdav",
    "/group/control_panel", "/api/", "/-/",
)
SKIP_EXTENSIONS = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff",
    ".woff2", ".ttf", ".eot", ".zip", ".mp4", ".webp", ".xml", ".json",
)
# Query strings on this portal are almost always session/rendering state, and
# following them multiplies the crawl without adding content.
SKIP_QUERY_KEYS = ("p_p_", "browserId", "minifierType", "themeId", "languageId", "t=")

# Links that leave the knowledge base: login walls, live chat, e-shop carts.
# Press releases, investor relations and awards are corporate-communications
# content, not customer service. They are also enormous — the press-release
# index alone is ~183k characters, which would have made roughly a fifth of the
# whole vector index press releases and diluted retrieval for every real
# customer question.
SKIP_PATH_CONTAINS = (
    "/login", "/live-chat", "signin", "logout",
    "press-release", "investor", "/awards", "/history", "/board-of-directors",
    "/te-museum", "/annual-report",
)

# Hard ceiling on a single document's contribution, as a second line of defence
# against one enormous page dominating the index.
MAX_DOC_CHARS = 40_000

PDF_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)


@dataclass
class Page:
    url: str
    status: int
    html: str
    depth: int


@dataclass
class CrawlStats:
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    by_lang: dict[str, int] = field(default_factory=dict)


def normalize_url(url: str) -> str:
    """Drop fragments and trailing slashes so /faq and /faq/ are one page."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"{parsed.scheme}://{netloc}{path}"


def is_crawlable(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    # Subdomains (ir.te.eg, csr.te.eg, my.te.eg) are separate properties and
    # mostly investor/login content — out of scope for a customer assistant.
    if host not in {h.replace("www.", "") for h in ALLOWED_HOSTS}:
        return False
    path = parsed.path.lower()
    if any(path.startswith(p) for p in SKIP_PATH_PREFIXES):
        return False
    if any(s in path for s in SKIP_PATH_CONTAINS):
        return False
    if path.endswith(SKIP_EXTENSIONS) or PDF_RE.search(url):
        return False
    if parsed.query and any(k in parsed.query for k in SKIP_QUERY_KEYS):
        return False
    return True


def extract_links(html: str, base_url: str) -> list[str]:
    tree = HTMLParser(html)
    out: list[str] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href")
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if is_crawlable(absolute):
            out.append(absolute)
    return out


def crawl(
    seeds: list[str],
    *,
    max_pages: int = 400,
    delay: float = 0.4,
    timeout: float = 25.0,
    max_depth: int = 3,
    seen: set[str] | None = None,
) -> tuple[list[Page], CrawlStats]:
    """Breadth-first crawl. Returns raw pages; cleaning happens in clean.py.

    Breadth-first matters here: te.eg's useful customer content (plans, FAQs,
    support) sits within two hops of the seeds, while depth-first would burrow
    into promo microsites.
    """
    stats = CrawlStats()
    # `seen` may be pre-seeded with URLs already held, so an append crawl skips
    # them without refetching. Seeds are queued regardless: a seed we already
    # hold is still needed as a source of links into its section.
    seen = set(seen or ())
    queue: deque[tuple[str, int]] = deque()

    for seed in seeds:
        norm = normalize_url(seed)
        seen.add(norm)
        queue.append((norm, 0))

    pages: list[Page] = []
    headers = {
        "User-Agent": USER_AGENT,
        # Ask for both languages; the portal decides per-URL anyway.
        "Accept-Language": "ar,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    }

    with httpx.Client(
        headers=headers, timeout=timeout, follow_redirects=True, http2=False
    ) as client:
        while queue and len(pages) < max_pages:
            url, depth = queue.popleft()
            try:
                response = client.get(url)
            except Exception as exc:  # network flake, DNS, timeout
                stats.failed += 1
                log.warning("fetch failed %s: %s", url, exc)
                continue

            if response.status_code != 200:
                stats.failed += 1
                continue
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                stats.skipped += 1
                continue

            html = response.text
            pages.append(Page(url=url, status=response.status_code, html=html, depth=depth))
            stats.fetched += 1
            if stats.fetched % 25 == 0:
                log.info("crawled %d/%d pages (queue=%d)", stats.fetched, max_pages, len(queue))

            if depth < max_depth:
                for link in extract_links(html, url):
                    if link not in seen:
                        seen.add(link)
                        queue.append((link, depth + 1))

            time.sleep(delay)

    return pages, stats

import logging
import os
from typing import Literal
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

SearchEngine = Literal["google", "ddg", "bing"]


def _block_google_enabled() -> bool:
    return os.environ.get("MCP_BLOCK_GOOGLE", "false").strip().lower() == "true"


def get_search_engine() -> SearchEngine:
    raw = os.environ.get("MCP_SEARCH_ENGINE", "ddg")
    engine = (raw or "ddg").strip().lower()
    if engine not in ("google", "ddg", "bing"):
        logger.warning(f"Invalid MCP_SEARCH_ENGINE='{raw}', defaulting to 'ddg'.")
        engine = "ddg"
    # If block is enabled and engine is google, we'll still return 'google' here,
    # but get_search_url will perform an actual fallback and warn.
    return engine  # type: ignore[return-value]


def get_search_url(query: str) -> str:
    engine = get_search_engine()
    block_google = _block_google_enabled()

    # Fallback if google is blocked
    effective_engine: SearchEngine = engine
    if engine == "google" and block_google:
        effective_engine = "ddg"
        logger.warning(
            "MCP_BLOCK_GOOGLE=true and MCP_SEARCH_ENGINE=google; falling back to DuckDuckGo for searches."
        )

    if effective_engine == "google":
        base = "https://www.google.com/search?q="
    elif effective_engine == "bing":
        base = "https://www.bing.com/search?q="
    else:  # ddg
        base = "https://duckduckgo.com/?q="

    from urllib.parse import quote_plus

    url = f"{base}{quote_plus(query)}"
    logger.info(f"Using search engine '{effective_engine}' for query: {query}")
    return url


def maybe_rewrite_blocked_url(url: str) -> str:
    """
    If MCP_BLOCK_GOOGLE=true and url is a Google search URL, rewrite to the configured
    engine (with google blocked fallback handled by get_search_url) using the 'q' param.
    Otherwise return unchanged.
    """
    if not _block_google_enabled():
        return url

    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()

        # Only consider google.* domains for search paths
        if "google." in host and path in ("/search", "/url"):
            qs = parse_qs(parsed.query or "")
            q_vals = qs.get("q") or []
            if q_vals:
                query = q_vals[0]
                rewritten = get_search_url(query)
                if rewritten != url:
                    logger.warning(f"Blocking Google URL; rewriting to: {rewritten}")
                return rewritten
    except Exception as e:
        logger.debug(f"maybe_rewrite_blocked_url: failed to parse or rewrite URL '{url}': {e}")

    return url

import logging
import os
from typing import Literal
from urllib.parse import urlparse, parse_qs, urlencode

logger = logging.getLogger(__name__)

SearchEngine = Literal["google", "ddg", "bing", "custom"]


def _block_google_enabled() -> bool:
    return os.environ.get("MCP_BLOCK_GOOGLE", "false").strip().lower() == "true"


def get_search_engine() -> SearchEngine:
    raw = os.environ.get("MCP_SEARCH_ENGINE", "bing")
    engine = (raw or "bing").strip().lower()
    if engine not in ("google", "ddg", "bing", "custom"):
        logger.warning(f"Invalid MCP_SEARCH_ENGINE='{raw}', defaulting to 'bing'.")
        engine = "bing"
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
    elif effective_engine == "ddg":
        base = "https://duckduckgo.com/?q="
    else:  # custom
        # Option A: Template that includes {q}
        template = os.environ.get("MCP_SEARCH_URL_TEMPLATE")
        if template:
            try:
                from urllib.parse import quote_plus
                url = template.replace("{q}", quote_plus(query))
                logger.info("Using custom search template for query.")
                return url
            except Exception as e:
                logger.warning(f"Failed to use MCP_SEARCH_URL_TEMPLATE, falling back to param mode: {e}")
        # Option B: Base URL + query param name
        base_url = os.environ.get("MCP_SEARCH_ENGINE_URL")
        q_param = os.environ.get("MCP_SEARCH_QUERY_PARAM", "q")
        if base_url:
            qs = urlencode({q_param: query})
            sep = '&' if ('?' in base_url) else '?'
            url = f"{base_url}{sep}{qs}"
            logger.info("Using custom search base URL with query param for query.")
            return url
        # If custom is selected but not configured, warn and default to ddg
        logger.warning("MCP_SEARCH_ENGINE=custom but no MCP_SEARCH_URL_TEMPLATE or MCP_SEARCH_ENGINE_URL provided; defaulting to DuckDuckGo.")
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

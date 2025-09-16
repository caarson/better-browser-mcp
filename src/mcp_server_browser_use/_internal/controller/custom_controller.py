import pdb
import json

import pyperclip
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from browser_use.controller.registry.service import Registry, RegisteredAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
import inspect
import asyncio
import os
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.agent.views import ActionModel, ActionResult

from ..utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools
from ..utils.search_engine import get_search_url, maybe_rewrite_blocked_url, get_search_engine

from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.mcp_client = None
        self.mcp_server_config = None
        # Track if the user closed the browser window to stop iteration fast
        self._user_closed_browser: bool = False

    async def _set_overlay(self, browser: BrowserContext, text: Optional[str]):
        """Inject or update a tiny overlay in the top-right corner of the active page."""
        try:
            pw_ctx = getattr(browser, "playwright_context", None)
            if not pw_ctx or not getattr(pw_ctx, "pages", None):
                return
            page = pw_ctx.pages[-1]
            await page.evaluate(
                """
                (text) => {
                    try {
                        let el = document.getElementById('__agent_overlay__');
                        if (!el) {
                            el = document.createElement('div');
                            el.id = '__agent_overlay__';
                            el.style.cssText = [
                                'position:fixed',
                                'top:8px',
                                'right:8px',
                                'z-index:2147483647',
                                'background:rgba(0,0,0,0.75)',
                                'color:#fff',
                                "font:12px/1.4 system-ui,Segoe UI,Arial,sans-serif",
                                'padding:6px 8px',
                                'border-radius:6px',
                                'pointer-events:none',
                                'max-width:30vw',
                                'box-shadow:0 2px 8px rgba(0,0,0,0.3)',
                                'white-space:nowrap',
                            ].join(';');
                            document.body.appendChild(el);
                        }
                        if (!text) {
                            el.style.display = 'none';
                            return;
                        }
                        el.style.display = 'block';
                        el.textContent = text;
                    } catch (e) { /* ignore */ }
                }
                """,
                text,
            )
        except Exception:
            # Never fail the action due to overlay errors
            pass

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. However, if you encounter a definitive blocker "
            "that prevents you from proceeding independently – such as needing credentials you don't possess, "
            "requiring subjective human judgment, needing a physical action performed, encountering complex CAPTCHAs, "
            "or facing limitations in your capabilities – you must request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                result = self.ask_assistant_callback(query, browser)
                user_response = await result if inspect.isawaitable(result) else result
                try:
                    resp_text = user_response.get('response') if isinstance(user_response, dict) else str(user_response)
                except Exception:
                    resp_text = str(user_response)
                msg = f"AI ask: {query}. User response: {resp_text}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(extracted_content="Human cannot help you. Please try another way.",
                                    include_in_memory=True)

        @self.registry.action(
            "Search documentation using site filters and open the results page."
        )
        async def doc_search(query: str, browser: BrowserContext, language: str | None = None, site: str | None = None, new_tab: bool = False):
            """
            Perform a documentation-focused search by applying site filters for official docs portals.
            Examples:
            - language="java" -> restrict to docs.oracle.com and javadoc.io
            - site="docs.oracle.com" -> restrict to that site only
            """
            q = query.strip()
            if not q:
                return ActionResult(error="Empty query for doc_search")

            filters = []
            if site:
                filters.append(f"site:{site}")
            elif language:
                lang = language.lower().strip()
                # Expand to common ecosystems
                if lang in ("java",):
                    filters.append("(site:docs.oracle.com OR site:javadoc.io OR site:hub.spigotmc.org)")
                elif lang in ("js", "javascript", "web"):
                    filters.append("site:developer.mozilla.org")
                elif lang in ("python",):
                    filters.append("(site:docs.python.org OR site:readthedocs.io)")
                elif lang in ("rust",):
                    filters.append("(site:docs.rs OR site:doc.rust-lang.org)")
                elif lang in ("go", "golang"):
                    filters.append("site:pkg.go.dev")
                elif lang in ("node", "nodejs"):
                    filters.append("site:nodejs.org/api")
                elif lang in ("dotnet", ".net", "csharp", "c#"):
                    filters.append("(site:learn.microsoft.com/en-us/dotnet/api OR site:learn.microsoft.com/en-us/aspnet)")
                elif lang in ("kotlin",):
                    filters.append("(site:kotlinlang.org/docs OR site:kotlinlang.org/api)")
                elif lang in ("swift",):
                    filters.append("site:developer.apple.com/documentation")

            filtered_query = f"{q} {' '.join(filters)}" if filters else q
            url = get_search_url(filtered_query)
            logger.info(f"doc_search -> {filtered_query} => {url}")
            # Reuse existing go_to_url machinery so URL rewriting hooks still apply
            return await self.registry.execute_action(
                "go_to_url",
                {"url": url, "new_tab": bool(new_tab)},
                browser=browser,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=None,
                context=None,
            )

        @self.registry.action(
            "Perform a documentation search, click the best documentation-site result, scroll to reveal content, detect profile, and extract sections. Returns structured JSON."
        )
        async def doc_orient_and_extract(query: str, browser: BrowserContext, language: str | None = None, site: str | None = None, scroll_times: int = 3):
            try:
                await self._set_overlay(browser, "researching…")
                # 1) Search
                res1 = await doc_search(query=query, browser=browser, language=language, site=site, new_tab=False)
                if isinstance(res1, ActionResult) and res1.error:
                    await self._set_overlay(browser, "")
                    return res1
                # 2) Click best doc-site result
                await self._set_overlay(browser, "opening doc…")
                res2 = await click_best_doc_result(browser=browser)
                if isinstance(res2, ActionResult) and res2.error:
                    # If no doc link found, stay on current page and try extraction anyway
                    pass
                # 3) Scroll to reveal content
                try:
                    await asyncio.sleep(0.3)
                except Exception:
                    pass
                await self._set_overlay(browser, "scrolling…")
                _ = await self.registry.execute_action("scroll_down", {"times": int(scroll_times)}, browser=browser, page_extraction_llm=None, sensitive_data=None, available_file_paths=None, context=None)
                # 4) Identify doc profile
                await self._set_overlay(browser, "profiling…")
                prof = await identify_doc_profile(browser=browser)
                prof_payload = {}
                if isinstance(prof, ActionResult) and prof.extracted_content:
                    try:
                        prof_payload = json.loads(prof.extracted_content)
                    except Exception:
                        prof_payload = {"raw": prof.extracted_content}
                # 5) Extract sections
                await self._set_overlay(browser, "reading content…")
                ext = await fetch_doc_sections_auto(browser=browser, max_items=200, include_html=False)
                ext_payload = {}
                if isinstance(ext, ActionResult) and ext.extracted_content:
                    try:
                        ext_payload = json.loads(ext.extracted_content)
                    except Exception:
                        ext_payload = {"raw": ext.extracted_content}
                payload = json.dumps({
                    "query": query,
                    "profile": prof_payload,
                    "extracted": ext_payload,
                }, ensure_ascii=False)
                await self._set_overlay(browser, "")
                return ActionResult(extracted_content=payload, include_in_memory=True)
            except Exception as e:
                try:
                    await self._set_overlay(browser, "")
                except Exception:
                    pass
                return ActionResult(error=f"doc_orient_and_extract failed: {e}")

        @self.registry.action(
            "Click the top documentation-site result on the current search page."
        )
        async def click_best_doc_result(
            browser: BrowserContext,
            allowed_sites: list[str] | None = None,
            include_github: bool = True,
            prefer_official: bool = True,
            new_tab: bool = False,
        ):
            """
            Heuristically select the best documentation result link from the current search results page and open it.
            Default domains include: docs.oracle.com, javadoc.io, developer.mozilla.org, docs.python.org, docs.rs,
            pkg.go.dev, nodejs.org/api, learn.microsoft.com (dotnet api), developer.apple.com/documentation,
            kotlinlang.org/docs or /api, readthedocs.io, dev.java, docs.gradle.org, docs.spring.io.
            GitHub is allowed (README/Wiki often serve as docs) when include_github=True.
            """
            try:
                pw_ctx = getattr(browser, 'playwright_context', None)
                if not pw_ctx:
                    return ActionResult(error="Playwright context missing")
                pages = getattr(pw_ctx, 'pages', None) or []
                if not pages:
                    return ActionResult(error="No open pages")
                page = pages[-1]

                default_allowed = {
                    "docs.oracle.com",
                    "javadoc.io",
                    "hub.spigotmc.org",
                    "developer.mozilla.org",
                    "docs.python.org",
                    "docs.rs",
                    "doc.rust-lang.org",
                    "pkg.go.dev",
                    "nodejs.org",
                    "learn.microsoft.com",
                    "developer.apple.com",
                    "kotlinlang.org",
                    "readthedocs.io",
                    "dev.java",
                    "docs.gradle.org",
                    "docs.spring.io",
                }
                if allowed_sites:
                    for s in allowed_sites:
                        if s and isinstance(s, str):
                            default_allowed.add(s.strip().lower())

                def _domain_score(host: str, path: str) -> int:
                    host_l = (host or "").lower()
                    path_l = (path or "").lower()
                    score = 0
                    # Official docs priority
                    if any(dom in host_l for dom in default_allowed):
                        score += 10
                    # API/doc path hints
                    if any(seg in path_l for seg in ["/api", "/docs", "/javadoc", "/javadocs", "/reference", "/documentation"]):
                        score += 4
                    # Node API special-case
                    if host_l.endswith("nodejs.org") and "/api" in path_l:
                        score += 6
                    # .NET API
                    if "learn.microsoft.com" in host_l and "/dotnet/" in path_l:
                        score += 6
                    return score

                def _github_score(host: str, path: str) -> int:
                    host_l = (host or "").lower()
                    path_l = (path or "").lower()
                    if host_l == "github.com":
                        # README/Wiki/docs folders commonly used as docs
                        s = 3
                        if any(seg in path_l for seg in ["/tree/", "/blob/", "/wiki", "/docs", "/readme"]):
                            s += 2
                        return s
                    return 0

                anchors = await page.query_selector_all('a[href]')
                from urllib.parse import urlparse, parse_qs
                candidates: list[tuple[int, str]] = []  # (score, url)

                for a in anchors:
                    href = await a.get_attribute('href')
                    if not href:
                        continue
                    href = href.strip()
                    if href.startswith('#') or href.lower().startswith('javascript:'):
                        continue
                    # Resolve relative URLs
                    try:
                        abs_url = await page.evaluate('(h) => new URL(h, window.location.href).href', href)
                    except Exception:
                        abs_url = href

                    try:
                        parsed = urlparse(abs_url)
                        host = (parsed.netloc or '').lower()
                        path = parsed.path or ''
                    except Exception:
                        continue

                    # Extract target from Google redirect links
                    if 'google.' in host and parsed.path.startswith('/url'):
                        qs = parse_qs(parsed.query or '')
                        q_vals = qs.get('q') or []
                        if q_vals:
                            abs_url = q_vals[0]
                            try:
                                parsed = urlparse(abs_url)
                                host = (parsed.netloc or '').lower()
                                path = parsed.path or ''
                            except Exception:
                                continue

                    # Skip non-target navigations and keep only doc-leaning links
                    doc_score = _domain_score(host, path)
                    gh_score = _github_score(host, path) if include_github else 0
                    total = doc_score + gh_score
                    if total <= 0:
                        continue

                    # Optional prefer_official: boost official non-GitHub over GitHub
                    if prefer_official and host != 'github.com' and doc_score >= 10:
                        total += 2

                    candidates.append((total, abs_url))

                if not candidates:
                    return ActionResult(error="No documentation-like links found on this page.")

                # Sort by score desc; keep order otherwise stable
                candidates.sort(key=lambda x: x[0], reverse=True)
                chosen_url = candidates[0][1]

                # Navigate via existing go_to_url action to keep consistency
                await self._set_overlay(browser, "opening doc…")
                return await self.registry.execute_action(
                    "go_to_url",
                    {"url": chosen_url, "new_tab": bool(new_tab)},
                    browser=browser,
                    page_extraction_llm=None,
                    sensitive_data=None,
                    available_file_paths=None,
                    context=None,
                )
            except Exception as e:
                return ActionResult(error=f"click_best_doc_result failed: {e}")

        @self.registry.action(
            "Open the Java SE API index on docs.oracle.com for quick class browsing."
        )
        async def open_java_api_index(browser: BrowserContext, version: str | None = None, new_tab: bool = False):
            """
            Open the official Java SE API index; defaults to a recent version if none provided.
            """
            ver = (version or "23").strip()  # default to a recent release
            # Construct Oracle API index URL
            base = f"https://docs.oracle.com/en/java/javase/{ver}/docs/api/index.html"
            logger.info(f"open_java_api_index -> {base}")
            await self._set_overlay(browser, "opening page…")
            return await self.registry.execute_action(
                "go_to_url",
                {"url": base, "new_tab": bool(new_tab)},
                browser=browser,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=None,
                context=None,
            )

        @self.registry.action(
            "Open javadoc.io search for libraries/classes by keyword."
        )
        async def open_javadoc_io_search(q: str, browser: BrowserContext, new_tab: bool = False):
            query = q.strip()
            if not query:
                return ActionResult(error="Empty query for open_javadoc_io_search")
            url = f"https://javadoc.io/search?q={query}"
            logger.info(f"open_javadoc_io_search -> {url}")
            await self._set_overlay(browser, "opening page…")
            return await self.registry.execute_action(
                "go_to_url",
                {"url": url, "new_tab": bool(new_tab)},
                browser=browser,
                page_extraction_llm=None,
                sensitive_data=None,
                available_file_paths=None,
                context=None,
            )

        @self.registry.action(
            "Extract main readable content from the current page (readability-like)."
        )
        async def extract_main_content(browser: BrowserContext):
            """
            Use the installed MainContentExtractor to pull the primary article/content area from the page.
            Returns plain text fallback if HTML extract is not available.
            """
            try:
                await self._set_overlay(browser, "reading content…")
                pw_ctx = getattr(browser, 'playwright_context', None)
                if not pw_ctx:
                    return ActionResult(error="Playwright context missing")
                pages = getattr(pw_ctx, 'pages', None) or []
                if not pages:
                    return ActionResult(error="No open pages")
                page = pages[-1]
                html = await page.content()
                extractor = MainContentExtractor()
                result = extractor.extract(html)
                text = result.text or ""
                if not text.strip():
                    # fallback: try to grab body innerText
                    try:
                        txt = await page.inner_text('body')
                        text = txt or ""
                    except Exception:
                        pass
                await self._set_overlay(browser, "")
                return ActionResult(extracted_content=text.strip(), include_in_memory=True)
            except Exception as e:
                try:
                    await self._set_overlay(browser, "")
                except Exception:
                    pass
                return ActionResult(error=f"extract_main_content failed: {e}")

        @self.registry.action(
            "Fetch structured sections from a Java javadoc page (class/method/field summaries)."
        )
        async def fetch_java_doc_sections(browser: BrowserContext, max_items: int = 100):
            """
            Heuristically extract common sections from Java docs (Oracle or javadoc.io):
            - headers: title, subtitle (package), page title
            - summaries: class, method, field, constructor
            - details sections
            Returns JSON with keys per section.
            """
            try:
                await self._set_overlay(browser, "reading content…")
                pw_ctx = getattr(browser, 'playwright_context', None)
                if not pw_ctx:
                    return ActionResult(error="Playwright context missing")
                pages = getattr(pw_ctx, 'pages', None) or []
                if not pages:
                    return ActionResult(error="No open pages")
                page = pages[-1]

                async def _text(sel: str) -> str:
                    try:
                        el = await page.query_selector(sel)
                        if not el:
                            return ""
                        return (await el.inner_text()).strip()
                    except Exception:
                        return ""

                async def _collect(selectors: list[str]) -> list[dict[str, str]]:
                    acc: list[dict[str, str]] = []
                    for sel in selectors:
                        try:
                            els = await page.query_selector_all(sel)
                            for el in els:
                                if len(acc) >= int(max_items):
                                    break
                                try:
                                    txt = (await el.inner_text()).strip()
                                    href = await el.get_attribute('href')
                                    abs_href = None
                                    if href:
                                        try:
                                            abs_href = await page.evaluate('(h) => new URL(h, window.location.href).href', href)
                                        except Exception:
                                            abs_href = href
                                    acc.append({"text": txt, **({"href": abs_href} if abs_href else {})})
                                except Exception:
                                    continue
                        except Exception:
                            continue
                    return acc

                data: dict[str, object] = {}
                data["page_title"] = await page.title()
                data["header_title"] = await _text(".header .title, h1.title, h1")
                data["header_subtitle"] = await _text(".header .subTitle, .subTitle")
                data["package"] = await _text(".header .subTitle, .package-signature .packageName")
                data["class_summary"] = await _collect(["#class-summary", "section.class-description", ".class-description"]) 
                data["constructor_summary"] = await _collect(["#constructor-summary", "table.constructor-summary, .constructor-summary"]) 
                data["method_summary"] = await _collect(["#method-summary", "table.method-summary, .method-summary, .memberSummary:has(> caption:has-text('Method Summary')) a"]) 
                data["field_summary"] = await _collect(["#field-summary", "table.field-summary, .field-summary, .memberSummary:has(> caption:has-text('Field Summary')) a"]) 
                data["details"] = await _collect(["#method-details, #field-details, #constructor-details, .details"]) 

                result = ActionResult(extracted_content=json.dumps(data, ensure_ascii=False), include_in_memory=True)
                try:
                    await self._set_overlay(browser, "")
                except Exception:
                    pass
                return result
            except Exception as e:
                try:
                    await self._set_overlay(browser, "")
                except Exception:
                    pass
                return ActionResult(error=f"fetch_java_doc_sections failed: {e}")

        @self.registry.action(
            "Identify the documentation profile for the current page based on the URL/domain."
        )
        async def identify_doc_profile(browser: BrowserContext):
            try:
                pw_ctx = getattr(browser, 'playwright_context', None)
                if not pw_ctx:
                    return ActionResult(error="Playwright context missing")
                pages = getattr(pw_ctx, 'pages', None) or []
                if not pages:
                    return ActionResult(error="No open pages")
                page = pages[-1]
                url = getattr(page, 'url', '') or ''

                profile = "generic"
                if "developer.mozilla.org" in url:
                    profile = "mdn"
                elif "docs.python.org" in url:
                    profile = "python_docs"
                elif "readthedocs.io" in url or "readthedocs" in url:
                    profile = "readthedocs"
                elif "docs.rs" in url or "doc.rust-lang.org" in url:
                    profile = "rust_docs"
                elif "pkg.go.dev" in url:
                    profile = "go_pkg"
                elif "nodejs.org" in url and "/api" in url:
                    profile = "node_api"
                elif "learn.microsoft.com" in url and "/dotnet/" in url:
                    profile = "dotnet_api"
                elif "docs.oracle.com" in url or "javadoc.io" in url or "hub.spigotmc.org" in url:
                    profile = "java_docs"
                elif "developer.apple.com/documentation" in url:
                    profile = "apple_docs"

                payload = json.dumps({"url": url, "profile": profile}, ensure_ascii=False)
                return ActionResult(extracted_content=payload, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"identify_doc_profile failed: {e}")

        @self.registry.action(
            "Auto-detect the documentation site and extract structured sections (headings, API items, anchors, and main content)."
        )
        async def fetch_doc_sections_auto(browser: BrowserContext, max_items: int = 200, include_html: bool = False):
            try:
                await self._set_overlay(browser, "reading content…")
                pw_ctx = getattr(browser, 'playwright_context', None)
                if not pw_ctx:
                    return ActionResult(error="Playwright context missing")
                pages = getattr(pw_ctx, 'pages', None) or []
                if not pages:
                    return ActionResult(error="No open pages")
                page = pages[-1]
                url = getattr(page, 'url', '') or ''

                async def _text(sel: str) -> str:
                    try:
                        el = await page.query_selector(sel)
                        if not el:
                            return ""
                        return (await el.inner_text()).strip()
                    except Exception:
                        return ""

                async def _collect_text(selectors: list[str], per: int = 50, with_href: bool = True) -> list[dict[str, str]]:
                    acc: list[dict[str, str]] = []
                    for sel in selectors:
                        try:
                            els = await page.query_selector_all(sel)
                            for el in els:
                                if len(acc) >= int(per):
                                    break
                                try:
                                    txt = (await el.inner_text()).strip()
                                    item: dict[str, str] = {"text": txt}
                                    if with_href:
                                        href = await el.get_attribute('href')
                                        if href:
                                            try:
                                                abs_href = await page.evaluate('(h) => new URL(h, window.location.href).href', href)
                                            except Exception:
                                                abs_href = href
                                            item["href"] = abs_href
                                    acc.append(item)
                                except Exception:
                                    continue
                        except Exception:
                            continue
                    return acc

                async def _collect_code(selectors: list[str], per: int = 50) -> list[str]:
                    acc: list[str] = []
                    for sel in selectors:
                        try:
                            els = await page.query_selector_all(sel)
                            for el in els:
                                if len(acc) >= int(per):
                                    break
                                try:
                                    code = await el.inner_text()
                                    if code and code.strip():
                                        acc.append(code.strip())
                                except Exception:
                                    continue
                        except Exception:
                            continue
                    return acc

                # Profile-specific extractors
                async def _extract_mdn() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("main h1, article h1, h1#content") ,
                        "headings": await _collect_text(["main h2, main h3, article h2, article h3"] , per=max_items),
                        "toc": await _collect_text(["nav.toc a, aside nav a"], per=max_items),
                        "code": await _collect_code(["pre code, code.hljs"], per=50),
                    }

                async def _extract_python_docs() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text(".document h1, #content h1, h1"),
                        "headings": await _collect_text([".document h2, .document h3, #content h2, #content h3"], per=max_items),
                        "toc": await _collect_text([".toc-tree a, .toctree-wrapper a, nav.bd-toc a, .sphinxsidebarwrapper a"], per=max_items),
                        "api_symbols": await _collect_text(["dl.class dt.sig a, dl.function dt.sig a, dt.sig a.reference"], per=max_items),
                        "code": await _collect_code(["div.highlight pre, pre code"], per=50),
                    }

                async def _extract_readthedocs() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text(".document h1, #content h1, h1"),
                        "headings": await _collect_text([".document h2, .document h3, #content h2, #content h3"], per=max_items),
                        "toc": await _collect_text(["nav.bd-toc a, .sphinxsidebar a, .wy-menu a, .toc-tree a"], per=max_items),
                        "code": await _collect_code(["div.highlight pre, pre code"], per=50),
                    }

                async def _extract_rust_docs() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("main h1, .fqn"),
                        "headings": await _collect_text(["main h2, main h3"], per=max_items),
                        "api_symbols": await _collect_text([".sidebar a.struct, .sidebar a.enum, .sidebar a.fn, .sidebar a.trait, .sidebar a.type"], per=max_items),
                        "code": await _collect_code(["pre code, code.hljs"], per=50),
                    }

                async def _extract_go_pkg() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("h1, #pkg-overview h1"),
                        "headings": await _collect_text(["#pkg-overview h2, #pkg-index h2, #pkg-functions h2, #pkg-types h2, main h2, main h3"], per=max_items),
                        "api_symbols": await _collect_text(["#pkg-index a, #pkg-functions a, #pkg-types a"], per=max_items),
                        "code": await _collect_code(["pre, pre code, .Code code"], per=50),
                    }

                async def _extract_node_api() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("main h1, .content h1, h1"),
                        "headings": await _collect_text(["main h2, main h3, .content h2, .content h3"], per=max_items),
                        "toc": await _collect_text(["nav.toc a, .toc a"], per=max_items),
                        "api_symbols": await _collect_text([".toc a[href*='#']"], per=max_items),
                        "code": await _collect_code(["pre code"], per=50),
                    }

                async def _extract_dotnet_api() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("article h1, h1"),
                        "headings": await _collect_text(["article h2, article h3"], per=max_items),
                        "toc": await _collect_text(["nav#affixed-left-container a, nav[aria-label='Table of contents'] a"], per=max_items),
                        "api_symbols": await _collect_text(["table a.symbol, .summary a, article a[href*='#']"], per=max_items),
                        "code": await _collect_code(["pre code, code.hljs"], per=50),
                    }

                async def _extract_java_docs() -> dict[str, object]:
                    # Reuse the java-specific selectors by calling the other action internally
                    try:
                        # emulate a call to fetch_java_doc_sections
                        res = await fetch_java_doc_sections(browser)  # type: ignore
                        if isinstance(res, ActionResult) and res.extracted_content:
                            return json.loads(res.extracted_content)
                    except Exception:
                        pass
                    # Fallback to generic if something goes wrong
                    return {}

                async def _extract_apple_docs() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("main h1, .documentation .title, h1"),
                        "headings": await _collect_text(["main h2, main h3, .documentation h2, .documentation h3"], per=max_items),
                        "toc": await _collect_text(["nav a[href*='#']"], per=max_items),
                        "code": await _collect_code(["pre code, code.hljs"], per=50),
                    }

                async def _extract_generic() -> dict[str, object]:
                    return {
                        "page_title": await page.title(),
                        "header_title": await _text("h1"),
                        "headings": await _collect_text(["h2, h3"], per=max_items),
                        "toc": await _collect_text(["nav a, aside a"], per=max_items),
                        "code": await _collect_code(["pre code"], per=50),
                    }

                # Decide profile
                profile = "generic"
                extractor = _extract_generic
                if "developer.mozilla.org" in url:
                    profile, extractor = "mdn", _extract_mdn
                elif "docs.python.org" in url:
                    profile, extractor = "python_docs", _extract_python_docs
                elif "readthedocs.io" in url or "readthedocs" in url:
                    profile, extractor = "readthedocs", _extract_readthedocs
                elif "docs.rs" in url or "doc.rust-lang.org" in url:
                    profile, extractor = "rust_docs", _extract_rust_docs
                elif "pkg.go.dev" in url:
                    profile, extractor = "go_pkg", _extract_go_pkg
                elif "nodejs.org" in url and "/api" in url:
                    profile, extractor = "node_api", _extract_node_api
                elif "learn.microsoft.com" in url and "/dotnet/" in url:
                    profile, extractor = "dotnet_api", _extract_dotnet_api
                elif "docs.oracle.com" in url or "javadoc.io" in url:
                    profile, extractor = "java_docs", _extract_java_docs
                elif "developer.apple.com/documentation" in url:
                    profile, extractor = "apple_docs", _extract_apple_docs

                data = await extractor()
                payload = json.dumps({
                    "url": url,
                    "profile": profile,
                    "data": data
                }, ensure_ascii=False)
                result = ActionResult(extracted_content=payload, include_in_memory=True)
                try:
                    await self._set_overlay(browser, "")
                except Exception:
                    pass
                return result
            except Exception as e:
                try:
                    await self._set_overlay(browser, "")
                except Exception:
                    pass
                return ActionResult(error=f"fetch_doc_sections_auto failed: {e}")

        @self.registry.action(
            "Fetch elements from the current page via CSS selectors and return their text or HTML."
        )
        async def fetch_elements(selectors: list[str], browser: BrowserContext, max_per_selector: int = 50, include_html: bool = False):
            """
            Extract structured content from documentation pages without excessive navigation.
            Returns a JSON object keyed by selector with ordered items.
            """
            try:
                pw_ctx = getattr(browser, 'playwright_context', None)
                if not pw_ctx:
                    return ActionResult(error="Playwright context missing")
                pages = getattr(pw_ctx, 'pages', None) or []
                if not pages:
                    return ActionResult(error="No open pages")
                page = pages[-1]

                out: dict[str, list[dict[str, str]]] = {}
                for sel in selectors or []:
                    try:
                        locs = await page.query_selector_all(sel)
                        items = []
                        for i, el in enumerate(locs):
                            if i >= int(max_per_selector):
                                break
                            try:
                                if include_html:
                                    content = await el.inner_html()
                                else:
                                    # Prefer innerText to avoid hidden text and scripts
                                    content = await el.inner_text()
                                href = None
                                try:
                                    href_prop = await el.get_attribute('href')
                                    href = href_prop if href_prop else None
                                except Exception:
                                    pass
                                items.append({"text": content.strip(), **({"href": href} if href else {})})
                            except Exception:
                                continue
                        out[sel] = items
                    except Exception:
                        out[sel] = []

                payload = json.dumps(out, ensure_ascii=False)
                return ActionResult(extracted_content=payload, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"fetch_elements failed: {e}")

        @self.registry.action(
            'Upload file to interactive element with file path ',
        )
        async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')

            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')

            dom_el = await browser.get_dom_element_by_index(index)

            file_upload_dom_el = dom_el.get_file_upload_element()

            if file_upload_dom_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)

            if file_upload_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: Optional[BrowserContext] = None,
            #
            page_extraction_llm: Optional[BaseChatModel] = None,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            #
            context: object | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    # Stop iterating if user closed the window (context or last page closed)
                    try:
                        # Attach close listeners once
                        def _ensure_close_watchers(browser: Optional[BrowserContext]):
                            try:
                                if not browser:
                                    return
                                pw_ctx = getattr(browser, 'playwright_context', None)
                                if not pw_ctx:
                                    return
                                if getattr(pw_ctx, '__bbmcp_close_watchers__', False):
                                    return
                                setattr(pw_ctx, '__bbmcp_close_watchers__', True)

                                def _mark_closed(*_args, **_kwargs):
                                    try:
                                        self._user_closed_browser = True
                                    except Exception:
                                        pass

                                try:
                                    pw_ctx.on('close', _mark_closed)
                                except Exception:
                                    pass

                                try:
                                    for p in getattr(pw_ctx, 'pages', []) or []:
                                        try:
                                            p.on('close', _mark_closed)
                                        except Exception:
                                            continue
                                except Exception:
                                    pass

                                try:
                                    pw_ctx.on('page', lambda pg: pg.on('close', _mark_closed))
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        _ensure_close_watchers(browser_context)
                        if self._user_closed_browser:
                            raise asyncio.CancelledError('Browser window closed by user')
                    except asyncio.CancelledError:
                        # Bubble up to stop the run
                        raise
                    except Exception:
                        # Ignore watcher setup errors
                        pass
                    # Set a small overlay to indicate current activity
                    try:
                        label_map = {
                            "search_google": "researching…",
                            "go_to_url": "opening page…",
                            "open_tab": "opening tab…",
                            "open_new_tab": "opening tab…",
                            "scroll_down": "scrolling…",
                            "scroll": "scrolling…",
                            "extract_main_content": "reading content…",
                            "fetch_doc_sections_auto": "reading content…",
                            "fetch_java_doc_sections": "reading content…",
                            "click_best_doc_result": "opening doc…",
                            "doc_search": "researching…",
                            "doc_orient_and_extract": "researching…",
                        }
                        label = label_map.get(action_name, f"{action_name.replace('_',' ')}…")
                        if browser_context:
                            await self._set_overlay(browser_context, label)
                    except Exception:
                        pass
                    # -- Search engine selection and URL rewrite hooks --
                    # Support both dict params and Pydantic BaseModel params
                    try:
                        from pydantic import BaseModel as _PydanticBaseModel
                    except Exception:
                        _PydanticBaseModel = tuple()  # type: ignore

                    def _get_attr(p, key, default=None):
                        if isinstance(p, dict):
                            return p.get(key, default)
                        return getattr(p, key, default)

                    def _set_attr(p, key, value):
                        if isinstance(p, dict):
                            p[key] = value
                            return p
                        setattr(p, key, value)
                        return p

                    # Handle search_google by converting to an engine-aware go_to_url
                    if action_name == "search_google":
                        query = _get_attr(params, "query") or _get_attr(params, "q")
                        if not query:
                            logger.warning("search_google called without 'query'. Passing through.")
                        else:
                            q = str(query)
                            ql = q.lower()
                            # Prefer official docs for Spigot/Bukkit queries to avoid unrelated results
                            if any(t in ql for t in ("spigot", "bukkit", "papermc", "spigotmc")) and "site:" not in ql:
                                q = f"{q} (site:hub.spigotmc.org OR site:javadoc.io)"
                            engine = get_search_engine()
                            url = get_search_url(q)
                            logger.info(f"search_google -> engine='{engine}', navigating to: {url}")
                            new_tab = bool(_get_attr(params, "new_tab", False))
                            action_name = "go_to_url"
                            if isinstance(params, _PydanticBaseModel):
                                params = GoToUrlAction(url=url, new_tab=new_tab)
                            else:
                                params = {"url": url, "new_tab": new_tab}

                    # For navigations, rewrite Google URLs to the selected engine as needed
                    if action_name in ("go_to_url", "open_tab", "open_new_tab"):
                        current_url = _get_attr(params, "url")
                        if current_url:
                            rewritten = maybe_rewrite_blocked_url(current_url)
                            if rewritten != current_url:
                                logger.info(f"Rewriting blocked URL: {current_url} -> {rewritten}")
                                params = _set_attr(params, "url", rewritten)

                    try:
                        if action_name.startswith("mcp"):
                            # this is a mcp tool
                            logger.debug(f"Invoke MCP tool: {action_name}")
                            mcp_tool = self.registry.registry.actions.get(action_name).function
                            result = await mcp_tool.ainvoke(params)
                        else:
                            result = await self.registry.execute_action(
                                action_name,
                                params,
                                browser=browser_context,
                                page_extraction_llm=page_extraction_llm,
                                sensitive_data=sensitive_data,
                                available_file_paths=available_file_paths,
                                context=context,
                            )
                    finally:
                        try:
                            if browser_context:
                                await self._set_overlay(browser_context, "")
                        except Exception:
                            pass

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        self.mcp_server_config = mcp_server_config
        if self.mcp_server_config:
            self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
            self.register_mcp_tools()

    def register_mcp_tools(self):
        """
        Register the MCP tools used by this controller.
        """
        if self.mcp_client:
            for server_name in self.mcp_client.server_name_to_tools:
                for tool in self.mcp_client.server_name_to_tools[server_name]:
                    tool_name = f"mcp.{server_name}.{tool.name}"
                    self.registry.registry.actions[tool_name] = RegisteredAction(
                        name=tool_name,
                        description=tool.description,
                        function=tool,
                        param_model=create_tool_param_model(tool),
                    )
                    logger.info(f"Add mcp tool: {tool_name}")

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)

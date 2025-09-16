import asyncio
import inspect
import json
import logging
import os
import traceback
import uuid
from typing import Any, Dict, Optional, Literal
from pathlib import Path
import sys


from .config import settings # Import global AppSettings instance

# Configure logging using settings
log_level_str = settings.server.logging_level.upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)

# Remove any existing handlers from the root logger to avoid duplicate messages
# if basicConfig was called elsewhere or by a library.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

_basic_cfg_kwargs: Dict[str, Any] = {
    "level": numeric_level,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "force": True,
}
if settings.server.log_file:
    _basic_cfg_kwargs["filename"] = settings.server.log_file
    _basic_cfg_kwargs["filemode"] = "a"
logging.basicConfig(**_basic_cfg_kwargs)

logger = logging.getLogger("mcp_server_browser_use")
# Prevent log propagation if other loggers are configured higher up
# logging.getLogger().propagate = False # This might be too aggressive, let's rely on basicConfig force

from browser_use.browser.browser import BrowserConfig
from mcp.server.fastmcp import Context, FastMCP

# Import from _internal
from ._internal.agent.browser_use.browser_use_agent import BrowserUseAgent
from ._internal.agent.deep_research.deep_research_agent import DeepResearchAgent
from ._internal.browser.custom_browser import CustomBrowser
from ._internal.browser.custom_context import (
    CustomBrowserContext,
    CustomBrowserContextConfig,
)
from ._internal.controller.custom_controller import CustomController
from ._internal.utils.search_engine import get_search_url
from ._internal.utils import llm_provider as internal_llm_provider # aliased

from browser_use.agent.views import (
    AgentHistoryList,
)

# Shared resources for MCP_BROWSER_KEEP_OPEN
shared_browser_instance: Optional[CustomBrowser] = None
shared_context_instance: Optional[CustomBrowserContext] = None
shared_controller_instance: Optional[CustomController] = None # Controller might also be shared
resource_lock = asyncio.Lock()


async def get_controller(ask_human_callback: Optional[Any] = None) -> CustomController:
    """Gets or creates a shared controller instance if keep_open is true, or a new one."""
    global shared_controller_instance
    if settings.browser.keep_open and shared_controller_instance:
        # Potentially update callback if it can change per call, though usually fixed for server
        return shared_controller_instance

    controller = CustomController(ask_assistant_callback=ask_human_callback)
    if settings.server.mcp_config:
        try:
            mcp_dict_config = settings.server.mcp_config
            if isinstance(settings.server.mcp_config, str):  # if passed as JSON string
                mcp_dict_config = json.loads(settings.server.mcp_config)
            await controller.setup_mcp_client(mcp_dict_config)
        except Exception as e:
            logger.error(f"Failed to setup MCP client for controller: {e}")

    if settings.browser.keep_open:
        shared_controller_instance = controller
    return controller


async def get_browser_and_context() -> tuple[CustomBrowser, CustomBrowserContext]:
    """
    Manages creation/reuse of CustomBrowser and CustomBrowserContext
    based on settings.browser.keep_open and settings.browser.use_own_browser.
    """
    global shared_browser_instance, shared_context_instance

    current_browser: Optional[CustomBrowser] = None
    current_context: Optional[CustomBrowserContext] = None

    agent_headless_override = settings.agent_tool.headless
    browser_headless = agent_headless_override if agent_headless_override is not None else settings.browser.headless

    agent_disable_security_override = settings.agent_tool.disable_security
    browser_disable_security = agent_disable_security_override if agent_disable_security_override is not None else settings.browser.disable_security

    if settings.browser.use_own_browser and settings.browser.cdp_url:
        logger.info(f"Connecting to own browser via CDP: {settings.browser.cdp_url}")
        browser_cfg = BrowserConfig(
            cdp_url=settings.browser.cdp_url,
            wss_url=settings.browser.wss_url,
            user_data_dir=settings.browser.user_data_dir, # Useful for CDP
            # Headless, binary_path etc. are controlled by the user-launched browser
        )
        current_browser = CustomBrowser(config=browser_cfg)
        # For CDP, context config is minimal, trace/recording might not apply or be harder to manage
        context_cfg = CustomBrowserContextConfig(
            trace_path=settings.browser.trace_path,
            save_downloads_path=settings.paths.downloads,
            save_recording_path=settings.agent_tool.save_recording_path if settings.agent_tool.enable_recording else None,
        )
        current_context = await current_browser.new_context(config=context_cfg)

    elif settings.browser.keep_open:
        if shared_browser_instance and shared_context_instance:
            logger.info("Reusing shared browser and context.")
            # Ensure browser is still connected
            if not shared_browser_instance.is_connected():
                logger.warning("Shared browser was disconnected. Recreating.")
                if shared_context_instance: await shared_context_instance.close() # Close old context too
                await shared_browser_instance.close() # Close browser after context
                shared_browser_instance = None
                shared_context_instance = None
            else:
                current_browser = shared_browser_instance
                # For shared browser, we might want a new context or reuse.
                # For simplicity, let's reuse the context if keep_open is true.
                # If new context per call is needed, this logic would change.
                current_context = shared_context_instance

                # Clean up any leftover initial blank page to avoid unused tabs.
                try:
                    for p in list(current_context.playwright_context.pages):
                        url = getattr(p, 'url', '') or ''
                        if url == 'about:blank':
                            try:
                                await p.close()
                            except Exception:
                                pass
                except Exception:
                    pass

        if not current_browser or not current_context : # If shared instances were not valid or not yet created
            logger.info("Creating new shared browser and context.")
            browser_cfg = BrowserConfig(
                headless=browser_headless,
                disable_security=browser_disable_security,
                browser_binary_path=settings.browser.binary_path,
                user_data_dir=settings.browser.user_data_dir,
                window_width=settings.browser.window_width,
                window_height=settings.browser.window_height,
            )
            shared_browser_instance = CustomBrowser(config=browser_cfg)
            context_cfg = CustomBrowserContextConfig(
                trace_path=settings.browser.trace_path,
                save_downloads_path=settings.paths.downloads,
                save_recording_path=settings.agent_tool.save_recording_path if settings.agent_tool.enable_recording else None,
                force_new_context=False # Important for shared context
            )
            shared_context_instance = await shared_browser_instance.new_context(config=context_cfg)
            current_browser = shared_browser_instance
            current_context = shared_context_instance
    else: # Create new resources per call (not using own browser, not keeping open)
        logger.info("Creating new browser and context for this call.")
        browser_cfg = BrowserConfig(
            headless=browser_headless,
            disable_security=browser_disable_security,
            browser_binary_path=settings.browser.binary_path,
            user_data_dir=settings.browser.user_data_dir,
            window_width=settings.browser.window_width,
            window_height=settings.browser.window_height,
        )
        current_browser = CustomBrowser(config=browser_cfg)
        context_cfg = CustomBrowserContextConfig(
            trace_path=settings.browser.trace_path,
            save_downloads_path=settings.paths.downloads,
            save_recording_path=settings.agent_tool.save_recording_path if settings.agent_tool.enable_recording else None,
            force_new_context=True
        )
        current_context = await current_browser.new_context(config=context_cfg)

    if not current_browser or not current_context:
        raise RuntimeError("Failed to initialize browser or context")

    return current_browser, current_context


def serve() -> FastMCP:
    server = FastMCP("mcp_server_browser_use")

    async def _documentation_quick_pipeline(query: str) -> str:
        """Run the controller's doc pipeline (doc_orient_and_extract) and summarize the result. Returns a short report string."""
        browser_instance: Optional[CustomBrowser] = None
        context_instance: Optional[CustomBrowserContext] = None
        controller_instance: Optional[CustomController] = None
        closed = False
        try:
            async with resource_lock:
                browser_instance, context_instance = await get_browser_and_context()
                controller_instance = await get_controller(ask_human_callback=None)
            if not context_instance or not controller_instance:
                return "Error: could not acquire browser/controller for documentation pipeline."

            # 1) Use the high-level doc action to search, click, scroll, detect profile, and extract sections (includes overlay updates)
            extracted_json: Optional[str] = None
            try:
                ql = (query or "").lower()
                lang = "java" if any(k in ql for k in ["spigot", "bukkit", "javadoc", "papermc"]) else None
                res = await controller_instance.registry.execute_action(
                    "doc_orient_and_extract",
                    {"query": query, "language": lang, "scroll_times": 3},
                    browser=context_instance,
                    page_extraction_llm=None,
                    sensitive_data=None,
                    available_file_paths=None,
                    context=None,
                )
                if hasattr(res, "extracted_content") and res.extracted_content:
                    extracted_json = str(res.extracted_content)
            except Exception as e:
                logger.warning(f"Quick doc pipeline: doc_orient_and_extract failed: {e}")
            # 2) Fallback to main content if structured extraction failed
            if not extracted_json:
                try:
                    res2 = await controller_instance.registry.execute_action(
                        "extract_main_content", {},
                        browser=context_instance,
                        page_extraction_llm=None,
                        sensitive_data=None,
                        available_file_paths=None,
                        context=None,
                    )
                    if hasattr(res2, "extracted_content") and res2.extracted_content:
                        extracted_json = str(res2.extracted_content)
                except Exception:
                    pass

            # 5) Summarize with LLM
            try:
                main_llm_config = settings.get_llm_config()
                llm = internal_llm_provider.get_llm_model(**main_llm_config)
                content_blob = extracted_json or ""
                if len(content_blob) > 18000:
                    content_blob = content_blob[:18000] + "\n… [truncated]"
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=(
                        "You are a documentation summarizer. Given a query and extracted sections from an official or authoritative documentation page, "
                        "produce a concise summary with 3–6 bullet key points and 2–4 direct links (anchors) to relevant sections if present. "
                        "Prefer API signatures and class/method names when available."
                    )),
                    HumanMessage(content=f"Query: {query}\n\nExtracted:\n{content_blob}")
                ]
                ai_msg = await llm.ainvoke(messages)
                return ai_msg.content if getattr(ai_msg, "content", None) else str(ai_msg)
            except Exception as e:
                logger.error(f"Quick doc pipeline summarization failed: {e}")
                return extracted_json or "No content extracted."
        finally:
            # Respect resource policy: if not keep_open and not use_own_browser, close after use
            try:
                if not settings.browser.keep_open and not settings.browser.use_own_browser:
                    closed = True
                    if context_instance:
                        await context_instance.close()
                    if browser_instance:
                        await browser_instance.close()
            except Exception:
                pass

    # Internal helper used by run_research/run_task; no longer exposed as an MCP tool
    async def run_browser_agent(ctx: Context, task: str, *, kickstart: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Received run_browser_agent task: {task[:100]}...")
        # Browsing rules: prefer lightweight completion and handle search engine auto-corrections
        search_rules = (
            "Preference: Complete the task directly in a single window/tab when feasible. Avoid spawning multiple tabs/windows unless required. "
            "Only escalate to a broader multi-source research approach if you encounter blockers or the task explicitly requires synthesis across many sources.\n"
            "Efficiency: Keep requests lean; perform at most 1–3 searches before reading and extracting from a likely best doc page.\n"
            "Completion: When you reach a conclusion, CALL THE 'done' ACTION with your final answer/summary so the session has a final result.\n\n"
            # Auto-correction handling across engines
            "Browsing rule: When a search results page shows text like 'Showing results for' and also offers 'Search instead for <literal>', "
            "click the 'Search instead for' (or equivalent) to force the exact query. This applies to Brave, Bing, DuckDuckGo, and Google. "
            "Also look for similar UI like 'Did you mean' or 'Including results for' and prefer exact-match links.\n\n"
        )
        task = f"{''.join(search_rules)}{task}"
        agent_task_id = str(uuid.uuid4())
        final_result = "Error: Agent execution failed."

        browser_instance: Optional[CustomBrowser] = None
        context_instance: Optional[CustomBrowserContext] = None
        controller_instance: Optional[CustomController] = None
        agent_instance: Optional[BrowserUseAgent] = None

        cancelled = False
        try:
            async with resource_lock:  # Protect shared resource access/creation
                browser_instance, context_instance = await get_browser_and_context()
                # For server, ask_human_callback is likely not interactive, can be None or a placeholder
                controller_instance = await get_controller(ask_human_callback=None)

            if not browser_instance or not context_instance or not controller_instance:
                raise RuntimeError("Failed to acquire browser resources or controller.")

            main_llm_config = settings.get_llm_config()
            main_llm = internal_llm_provider.get_llm_model(**main_llm_config)

            planner_llm = None
            if settings.llm.planner_provider and settings.llm.planner_model_name:
                planner_llm_config = settings.get_llm_config(is_planner=True)
                planner_llm = internal_llm_provider.get_llm_model(**planner_llm_config)

            agent_history_json_file = None
            task_history_base_path = settings.agent_tool.history_path

            if task_history_base_path:
                task_specific_history_dir = Path(task_history_base_path) / agent_task_id
                task_specific_history_dir.mkdir(parents=True, exist_ok=True)
                agent_history_json_file = str(task_specific_history_dir / f"{agent_task_id}.json")
                logger.info(f"Agent history will be saved to: {agent_history_json_file}")

            # Optional pre-navigation to ensure a page opens (e.g., search results)
            try:
                if kickstart and context_instance and controller_instance:
                    ks_kind = str(kickstart.get("kind", "")).lower()
                    ks_query = str(kickstart.get("query", ""))
                    if ks_kind in {"search", "doc", "doc_orient"} and ks_query:
                        url = get_search_url(ks_query)
                        logger.info(f"Kickstart navigation to search results: {url}")
                        # Use controller registry to navigate so URL rewriting hooks apply
                        await controller_instance.registry.execute_action(
                            "go_to_url",
                            {"url": url, "new_tab": False},
                            browser=context_instance,
                            page_extraction_llm=None,
                            sensitive_data=None,
                            available_file_paths=None,
                            context=None,
                        )
            except Exception as e:
                logger.warning(f"Kickstart navigation failed (continuing without): {e}")

            agent_instance = BrowserUseAgent(
                task=task,
                llm=main_llm,
                browser=browser_instance,
                browser_context=context_instance,
                controller=controller_instance,
                planner_llm=planner_llm,
                max_actions_per_step=settings.agent_tool.max_actions_per_step,
                use_vision=settings.agent_tool.use_vision,
            )

            history: AgentHistoryList = await agent_instance.run(max_steps=settings.agent_tool.max_steps)

            if agent_history_json_file:
                agent_instance.save_history(agent_history_json_file)

            final_result = history.final_result() or "Agent finished without a final result."
            logger.info(f"Agent task completed. Result: {final_result[:100]}...")

        except asyncio.CancelledError:
            # Propagate cancellation to the agent and ensure browser is stopped
            logger.warning("run_browser_agent was cancelled by the client; stopping agent and closing browser/context...")
            try:
                if agent_instance and hasattr(agent_instance, 'stop'):
                    maybe = agent_instance.stop()
                    if inspect.isawaitable(maybe):
                        await maybe
                elif agent_instance and hasattr(agent_instance, 'state'):
                    try:
                        setattr(agent_instance.state, 'stopped', True)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error while stopping agent on cancel: {e}")

            # Force-close resources on cancel even if keep_open/use_own_browser
            orig_context = context_instance
            orig_browser = browser_instance
            try:
                if orig_context:
                    await orig_context.close()
            except Exception:
                pass
            try:
                if orig_browser:
                    await orig_browser.close()
            except Exception:
                pass

            # Null out references to avoid double close in finally
            context_instance = None
            browser_instance = None

            # Clear shared instances if they were pointing to the ones we closed
            try:
                if settings.browser.keep_open:
                    global shared_browser_instance, shared_context_instance
                    # Compare against original references before we nulled locals
                    if shared_context_instance is orig_context:
                        shared_context_instance = None
                    if shared_browser_instance is orig_browser:
                        shared_browser_instance = None
            except Exception:
                pass

            # Try to close controller MCP client if it's not shared
            try:
                if controller_instance and not (settings.browser.keep_open and controller_instance == shared_controller_instance):
                    await controller_instance.close_mcp_client()
            except Exception:
                pass

            final_result = "Cancelled by client."
            cancelled = True
            # After cleanup, return a cancelled result
            return final_result
        except Exception as e:
            logger.error(f"Error in run_browser_agent: {e}\n{traceback.format_exc()}")
            final_result = f"Error: {e}"
        finally:
            # Normal cleanup when not cancelled
            if not cancelled and not settings.browser.keep_open and not settings.browser.use_own_browser:
                try:
                    logger.info("Closing browser resources for this call.")
                    if context_instance:
                        await context_instance.close()
                    if browser_instance:
                        await browser_instance.close()
                finally:
                    if controller_instance:  # Close controller only if not shared
                        try:
                            await controller_instance.close_mcp_client()
                        except Exception:
                            pass
            elif not cancelled and settings.browser.use_own_browser:  # Own browser, only close controller if not shared
                if controller_instance and not (settings.browser.keep_open and controller_instance == shared_controller_instance):
                    try:
                        await controller_instance.close_mcp_client()
                    except Exception:
                        pass
        return final_result

    @server.tool()
    async def run_deep_research(
        ctx: Context,
        research_task: str,
        max_windows: Optional[int] = None,
    ) -> str:
        logger.info(f"Received run_deep_research task: {research_task[:100]}...")
        task_id = str(uuid.uuid4()) # This task_id is used for the sub-directory name
        report_content = "Error: Deep research failed."
        agent_instance: Optional[DeepResearchAgent] = None

        try:
            main_llm_config = settings.get_llm_config() # Deep research uses main LLM config
            research_llm = internal_llm_provider.get_llm_model(**main_llm_config)

            # Prepare browser_config dict for DeepResearchAgent's sub-agents
            dr_browser_cfg = {
                "headless": settings.browser.headless, # Use general browser headless for sub-tasks
                "disable_security": settings.browser.disable_security,
                "browser_binary_path": settings.browser.binary_path,
                "user_data_dir": settings.browser.user_data_dir,
                "window_width": settings.browser.window_width,
                "window_height": settings.browser.window_height,
                "trace_path": settings.browser.trace_path, # For sub-agent traces
                "save_downloads_path": settings.paths.downloads, # For sub-agent downloads
            }
            if settings.browser.use_own_browser and settings.browser.cdp_url:
                # If main browser is CDP, sub-agents should also use it
                dr_browser_cfg["cdp_url"] = settings.browser.cdp_url
                dr_browser_cfg["wss_url"] = settings.browser.wss_url

            mcp_server_config_for_agent = None
            if settings.server.mcp_config:
                mcp_server_config_for_agent = settings.server.mcp_config
                if isinstance(settings.server.mcp_config, str):
                     mcp_server_config_for_agent = json.loads(settings.server.mcp_config)

            agent_instance = DeepResearchAgent(
                llm=research_llm,
                browser_config=dr_browser_cfg,
                mcp_server_config=mcp_server_config_for_agent,
            )

            current_max_parallel_browsers = max_windows if max_windows is not None else settings.research_tool.max_windows

            # Check if save_dir is provided, otherwise use in-memory approach
            save_dir_for_this_task = None
            if settings.research_tool.save_dir:
                # If save_dir is provided, construct the full save directory path for this specific task
                save_dir_for_this_task = str(Path(settings.research_tool.save_dir) / task_id)
                logger.info(f"Deep research save directory for this task: {save_dir_for_this_task}")
            else:
                logger.info("No save_dir configured. Deep research will operate in memory-only mode.")

            logger.info(f"Using max_windows: {current_max_parallel_browsers}")

            result_dict = await agent_instance.run(
                topic=research_task,
                save_dir=save_dir_for_this_task, # Can be None now
                task_id=task_id, # Pass the generated task_id
                max_windows=current_max_parallel_browsers
            )

            # Handle the result based on if files were saved or not
            if save_dir_for_this_task and result_dict.get("report_file_path") and Path(result_dict["report_file_path"]).exists():
                with open(result_dict["report_file_path"], "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                report_content = f"Deep research report generated successfully at {result_dict['report_file_path']}\n\n{markdown_content}"
                logger.info(f"Deep research task {task_id} completed. Report at {result_dict['report_file_path']}")
            elif result_dict.get("status") == "completed" and result_dict.get("final_report"):
                report_content = f"Deep research completed. Report content:\n\n{result_dict['final_report']}"
                if result_dict.get("report_file_path"):
                     report_content += f"\n(Expected report file at: {result_dict['report_file_path']})"
                logger.info(f"Deep research task {task_id} completed. Report content retrieved directly.")
            else:
                report_content = f"Deep research task {task_id} result: {result_dict}. Report file not found or content not available."
                logger.warning(report_content)


        except asyncio.CancelledError:
            logger.warning("run_deep_research was cancelled by the client; stopping agent...")
            try:
                if agent_instance and hasattr(agent_instance, 'stop'):
                    maybe = agent_instance.stop()
                    if inspect.isawaitable(maybe):
                        await maybe
            except Exception as e:
                logger.error(f"Error while stopping deep research agent on cancel: {e}")
            return "Cancelled by client."
        except Exception as e:
            logger.error(f"Error in run_deep_research: {e}\n{traceback.format_exc()}")
            report_content = f"Error: {e}"

        return report_content

    @server.tool()
    async def run_task(
        ctx: Context,
        task: str,
        max_windows: Optional[int] = None,
    ) -> str:
        """
        Smart router that prefers the lightweight browser task and escalates to deep research only when needed.

        Routing modes (env MCP_TASK_ROUTER_MODE):
        - auto (default): heuristics choose between task and deep_research
        - always-task: always use run_browser_agent
        - always-research: always use run_deep_research
        """
        mode = os.getenv("MCP_TASK_ROUTER_MODE", "auto").lower()
        logger.info(f"Router received task (mode={mode}): {task[:100]}...")

        def needs_deep_research(text: str) -> bool:
            t = text.lower()
            simple_patterns = [
                "open github", "go to github", "open readme", "open documentation", "read docs", "install guide",
                "open website", "navigate to", "visit https://", "go to https://", "open url",
                "search for", "find repo", "open repo"
            ]
            complex_markers = [
                "write a report", "compare", "synthesize", "summarize across", "survey", "state of the art",
                "multiple sources", "gather sources", "citation", "references", "deep research", "investigate",
                "root cause", "troubleshoot", "doesn't work", "error log", "fix bug", "unknown issue",
            ]
            # If explicitly mentions deep research or multi-source synthesis, escalate
            if any(p in t for p in complex_markers):
                return True
            # If it looks like straightforward navigation/search, keep it light
            if any(p in t for p in simple_patterns):
                return False
            # Fallback: short tasks likely simple, long descriptive prompts more likely research
            return len(t) > 240

        if mode == "always-task":
            return await run_browser_agent(ctx, task)  # type: ignore
        if mode == "always-research":
            return await run_deep_research(ctx, task, max_windows)  # type: ignore

        # auto mode
        if needs_deep_research(task):
            logger.info("Router: escalating to deep research based on heuristic.")
            return await run_deep_research(ctx, task, max_windows)  # type: ignore
        else:
            logger.info("Router: using lightweight task (browser agent).")
            return await run_browser_agent(ctx, task)  # type: ignore

    @server.tool()
    async def run_research(
        ctx: Context,
        topic_or_task: str,
        mode: Literal["auto", "task", "research", "documentation", "deep_research"] = "auto",
        max_windows: Optional[int] = None,
    ) -> str:
        """
    Unified entry point for browser work with modes:
    - auto (default): choose between task, research (lightweight), documentation, and deep_research
    - task: concrete UI/browser actions (e.g., Cloudflare DNS, consoles)
    - research: lightweight research/summarization using a single browser agent
    - documentation: documentation-first navigation and summarization (prefers official docs, API refs)
    - deep_research: full deep research pipeline

    Env override: MCP_RESEARCH_MODE=auto|task|research|documentation|deep_research
        """
        env_mode = os.getenv("MCP_RESEARCH_MODE", "auto").lower()
        # If caller explicitly picks a mode, honor it; if left at default 'auto', allow env to set default mode
        effective_mode = env_mode if mode == "auto" and env_mode in {"auto", "task", "research", "documentation", "deep_research"} else mode
        text = topic_or_task or ""

        logger.info(f"run_research received (mode={effective_mode}): {text[:120]}...")

        def needs_deep_research(t: str) -> bool:
            s = t.lower()
            complex_markers = [
                "write a report", "compare", "synthesize", "summarize across", "survey", "state of the art",
                "multiple sources", "gather sources", "citation", "references", "deep research", "investigate",
                "root cause", "troubleshoot", "doesn't work", "error log", "fix bug", "unknown issue",
            ]
            if any(p in s for p in complex_markers):
                return True
            return len(s) > 300

        def looks_like_task(t: str) -> bool:
            s = t.lower()
            task_markers = [
                "open ", "navigate to", "go to", "visit ", "click ", "log in", "login", "sign in",
                "dashboard", "console", "settings", "configure", "enable", "disable", "create record",
                "add record", "dns", "cloudflare", "github actions", "pipeline", "repo settings",
                "fill", "form", "upload", "download", "submit", "delete", "remove", "rename",
            ]
            return any(p in s for p in task_markers)

        def looks_like_research(t: str) -> bool:
            s = t.lower()
            research_markers = [
                "what is", "how does", "why is", "explain", "overview", "pros and cons", "alternatives",
                "tutorial", "examples", "guide", "docs", "documentation", "latest", "news", "compare",
                "benchmark", "performance", "advantages", "disadvantages", "roadmap",
            ]
            return any(p in s for p in research_markers)

        def looks_like_documentation(t: str) -> bool:
            s = t.lower()
            doc_markers = [
                "api reference", "reference docs", "javadoc", "java doc", "oracle docs", "javadoc.io",
                "class ", "method ", "package ", "interface ", "mdn", "python docs", "pypi docs",
                "read the docs", "rtd", "godoc", "pkg.go.dev", "rust docs", "docs.rs", "nuget docs",
                "dotnet api", "typescript api", "js api", "kotlin docs", "swift docs", "objc docs",
                # Heuristics: general documentation intents and Java ecosystem keywords
                "documentation", "docs", "api docs", "spigot", "bukkit", "papermc",
            ]
            return any(p in s for p in doc_markers)

        # Honor explicit modes first
        if effective_mode in {"task", "research", "documentation", "deep_research"}:
            chosen_mode = effective_mode
        else:
            # auto routing
            if needs_deep_research(text):
                chosen_mode = "deep_research"
            elif looks_like_documentation(text):
                chosen_mode = "documentation"
            elif looks_like_task(text):
                chosen_mode = "task"
            elif looks_like_research(text):
                chosen_mode = "research"
            else:
                # default to lightweight task to err on the side of minimal resource usage
                chosen_mode = "task"

        logger.info(f"run_research routing to: {chosen_mode}")

        if chosen_mode == "deep_research":
            return await run_deep_research(ctx, text, max_windows)  # type: ignore

        # Lightweight path uses run_browser_agent with a small, mode-specific prefix
        if chosen_mode == "task":
            prefix = (
                "Mode: TASK. Perform the concrete browser actions requested (navigation, forms, toggles, settings). "
                "Favor a single window/tab; act directly and stop when the task is done.\n\n"
            )
        elif chosen_mode == "documentation":
            prefix = (
                "Mode: DOCUMENTATION. Begin with a brief plan: identify target tech (library/runtime), scope (API/class/package), and 2–3 concise queries. "
                "Prefer official documentation and API references; keep requests efficient and tabs minimal.\n"
                "- First action: call doc_orient_and_extract with the EXACT topic_or_task. If it mentions Java/Javadoc/Spigot/Bukkit/Paper, pass language='java'.\n"
                "- Prefer: oracle.com docs, javadoc.io, developer.oracle.com, developer.mozilla.org, docs.python.org, docs.rs.\n"
                "- Doc-site detection heuristics: look for 'Packages/Classes/Index' (Java), API signature blocks, breadcrumbs, and the domains above.\n"
                "- If Java-related (mentions class/package/interface or 'javadoc'): try javadoc.io search or the Oracle Java SE API index; otherwise run a targeted doc search.\n"
                "- Use doc actions when appropriate: doc_search, doc_orient_and_extract, click_best_doc_result, open_java_api_index, open_javadoc_io_search, identify_doc_profile, fetch_doc_sections_auto, extract_main_content, fetch_java_doc_sections, scroll_down, collect_doc_overview, open_anchor_by_text.\n"
                "- On search result pages that say 'Showing results for' and offer 'Search instead for <literal>', click the exact-match link.\n"
                "- Keep a single window; open at most one extra tab for search when needed, then return to the main doc tab. Avoid logins or account changes.\n"
                "- After landing on a doc page: scroll 2–3 times (scroll_down) to reveal content; then run identify_doc_profile. If profile=java_docs, use fetch_java_doc_sections; otherwise use fetch_doc_sections_auto and/or extract_main_content.\n"
                "  On Oracle Java API index pages, navigate via Packages/Classes links to the relevant class or method section before summarizing.\n"
                "- Finish with a FINAL SUMMARY containing: page title, 1–3 key points, relevant API signatures, and direct links (anchors) to sections.\n\n"
            )
        else:  # research
            prefix = (
                "Mode: RESEARCH. Find information, read carefully, and summarize the key findings with links to sources. "
                "Prefer minimal tabs; do not change account settings or perform account logins.\n"
                "- You must browse the web for this task. FIRST ACTION: call search_google with the EXACT topic_or_task and open the results page.\n"
                "- After landing on results, open the most relevant official documentation or authoritative source and extract key points before summarizing.\n\n"
            )

        # Execute the lightweight agent once; if it ends without a final result, retry in documentation mode
        kickstart: Optional[Dict[str, Any]] = None
        # Only auto-open a SERP when explicitly enabled; otherwise let the agent formulate the query
        import os as _os
        if _os.getenv("MCP_KICKSTART_SEARCH", "false").lower() in {"1", "true", "yes"}:
            if chosen_mode in {"research", "documentation"}:
                kickstart = {"kind": "search", "query": text}
        result = await run_browser_agent(ctx, f"{prefix}{text}", kickstart=kickstart)  # type: ignore

        if (
            result.strip().lower().startswith("agent finished without a final result")
            and chosen_mode in {"research", "task"}
        ):
            logger.info("Lightweight run yielded no final result; retrying in documentation mode for better coverage.")
            doc_prefix = (
                "Mode: DOCUMENTATION. Begin with a brief plan: identify target tech (library/runtime), scope (API/class/package), and 2–3 concise queries. "
                "Prefer official documentation and API references; keep requests efficient and tabs minimal.\n"
                "- First action: call doc_orient_and_extract with the EXACT topic_or_task. If it mentions Java/Javadoc/Spigot/Bukkit/Paper, pass language='java'.\n"
                "- Prefer: oracle.com docs, javadoc.io, developer.oracle.com, developer.mozilla.org, docs.python.org, docs.rs.\n"
                "- Doc-site detection heuristics: look for 'Packages/Classes/Index' (Java), API signature blocks, breadcrumbs, and the domains above.\n"
                "- If Java-related (mentions class/package/interface or 'javadoc'): try javadoc.io search or the Oracle Java SE API index; otherwise run a targeted doc search.\n"
                "- Use doc actions when appropriate: doc_search, doc_orient_and_extract, click_best_doc_result, open_java_api_index, open_javadoc_io_search, identify_doc_profile, fetch_doc_sections_auto, extract_main_content, fetch_java_doc_sections, scroll_down, collect_doc_overview, open_anchor_by_text.\n"
                "- On search result pages that say 'Showing results for' and offer 'Search instead for <literal>', click the exact-match link.\n"
                "- Keep a single window; open at most one extra tab for search when needed, then return to the main doc tab. Avoid logins or account changes.\n"
                "- After landing on a doc page: scroll 2–3 times (scroll_down) to reveal content; then run identify_doc_profile. If profile=java_docs, use fetch_java_doc_sections; otherwise use fetch_doc_sections_auto and/or extract_main_content.\n"
                "  On Oracle Java API index pages, navigate via Packages/Classes links to the relevant class or method section before summarizing.\n"
                "- Finish with a FINAL SUMMARY containing: page title, 1–3 key points, relevant API signatures, and direct links (anchors) to sections.\n\n"
            )
            _ks = None
            try:
                import os as __os
                if __os.getenv("MCP_KICKSTART_SEARCH", "false").lower() in {"1", "true", "yes"}:
                    _ks = {"kind": "search", "query": text}
            except Exception:
                _ks = None
            result = await run_browser_agent(ctx, f"{doc_prefix}{text}", kickstart=_ks)  # type: ignore

        # Final safeguard: if still no result, run the controller-driven documentation pipeline
        if result.strip().lower().startswith("agent finished without a final result"):
            logger.info("Agent returned no final result after retry; running quick documentation pipeline.")
            try:
                fallback = await _documentation_quick_pipeline(text)
                if fallback and isinstance(fallback, str) and fallback.strip():
                    return fallback
            except Exception as e:
                logger.error(f"Documentation fallback failed: {e}")

        return result

    @server.tool()
    async def run_auto(
        ctx: Context,
        topic_or_task: str,
        max_windows: Optional[int] = None,
    ) -> str:
        """
        Auto-select between task, research, and deep_research based on heuristics.
        This is equivalent to calling run_research(..., mode="auto").
        """
        return await run_research(ctx, topic_or_task, mode="auto", max_windows=max_windows)  # type: ignore

    @server.tool()
    async def run_documentation(
        ctx: Context,
        topic_or_task: str,
    ) -> str:
        """
        Documentation-focused browsing: prefers official docs and API references and summarizes clearly.
        Equivalent to run_research(..., mode="documentation").
        """
        return await run_research(ctx, topic_or_task, mode="documentation")  # type: ignore

    return server

server_instance = serve() # Renamed from 'server' to avoid conflict with 'settings.server'

def main():
    # Respect help/version probes so clients like uvx or MCP tool discovery don't accidentally start the server
    args = sys.argv[1:]
    if any(a in ("-h", "--help", "help", "/?") for a in args):
        # Print a compact usage and exit without starting the server
        try:
            from importlib.metadata import version
            ver = version("mcp_server_browser_use")
        except Exception:
            ver = "unknown"
        usage = (
            f"mcp-server-browser-use v{ver}\n"
            "Usage: mcp-server-browser-use [--help] [--version]\n\n"
            "Starts the MCP server on stdio. Configure via environment variables.\n\n"
            "Exposed MCP tools:\n"
            "  - run_auto(topic_or_task, max_windows?)\n"
            "  - run_research(topic_or_task, mode=auto|task|research|documentation|deep_research, max_windows?)\n"
            "  - run_task(task, max_windows?)\n"
            "  - run_deep_research(research_task, max_windows?)\n"
            "  - run_documentation(topic_or_task)\n"
        )
        print(usage)
        return
    if any(a in ("-V", "--version", "version") for a in args):
        try:
            from importlib.metadata import version
            ver = version("mcp_server_browser_use")
        except Exception:
            ver = "unknown"
        print(ver)
        return

    logger.info("Starting MCP server for browser-use...")
    try:
        # Just log the Research tool save directory if it's configured
        if settings.research_tool.save_dir:
            logger.info(f"Research tool save directory configured: {settings.research_tool.save_dir}")
        else:
            logger.info("Research tool save directory not configured. Deep research will operate in memory-only mode.")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return # Exit if there's a configuration error

    logger.info(f"Loaded settings with LLM provider: {settings.llm.provider}, Model: {settings.llm.model_name}")
    logger.info(f"Browser keep_open: {settings.browser.keep_open}, Use own browser: {settings.browser.use_own_browser}")
    if settings.browser.use_own_browser:
        logger.info(f"Connecting to own browser via CDP: {settings.browser.cdp_url}")
    # Print a brief startup notice to stderr for interactive runs
    try:
        print("mcp-server-browser-use: running (waiting for MCP client on stdio)...", file=sys.stderr, flush=True)
    except Exception:
        pass
    server_instance.run()

if __name__ == "__main__":
    main()

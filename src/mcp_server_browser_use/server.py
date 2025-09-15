import asyncio
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
from ._internal.agent.task_agent import TaskAgent
from ._internal.agent.research_agent import ResearchAgent
from ._internal.browser.custom_browser import CustomBrowser
from ._internal.browser.custom_context import (
    CustomBrowserContext,
    CustomBrowserContextConfig,
)
from ._internal.controller.custom_controller import CustomController
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
            if isinstance(settings.server.mcp_config, str): # if passed as JSON string
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

    # Internal helper used by run_research/run_task; no longer exposed as an MCP tool
    async def run_browser_agent(ctx: Context, task: str) -> str:
        logger.info(f"Received run_browser_agent task: {task[:100]}...")
        # Browsing rules: prefer lightweight completion and handle search engine auto-corrections
        search_rules = (
            "Preference: Complete the task directly in a single window/tab when feasible. Avoid spawning multiple tabs/windows unless required. "
            "Only escalate to a broader multi-source research approach if you encounter blockers or the task explicitly requires synthesis across many sources.\n\n"
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

        try:
            async with resource_lock: # Protect shared resource access/creation
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

            agent_instance = TaskAgent(
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

        except Exception as e:
            logger.error(f"Error in run_browser_agent: {e}\n{traceback.format_exc()}")
            final_result = f"Error: {e}"
        finally:
            if not settings.browser.keep_open and not settings.browser.use_own_browser:
                logger.info("Closing browser resources for this call.")
                if context_instance:
                    await context_instance.close()
                if browser_instance:
                    await browser_instance.close()
                if controller_instance: # Close controller only if not shared
                    await controller_instance.close_mcp_client()
            elif settings.browser.use_own_browser: # Own browser, only close controller if not shared
                 if controller_instance and not (settings.browser.keep_open and controller_instance == shared_controller_instance):
                    await controller_instance.close_mcp_client()
        return final_result

    @server.tool()
    async def run_deep_research(
        ctx: Context,
        research_task: str,
        max_tabs: Optional[int] = None,
        max_windows: Optional[int] = None,  # deprecated alias
    ) -> str:
        logger.info(f"Received run_deep_research task: {research_task[:100]}...")
        task_id = str(uuid.uuid4()) # This task_id is used for the sub-directory name
        report_content = "Error: Deep research failed."

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

            agent_instance = ResearchAgent(
                llm=research_llm,
                browser_config=dr_browser_cfg,
                mcp_server_config=mcp_server_config_for_agent,
            )

            # Backward compatible selection: prefer explicit max_tabs, then legacy max_windows param, then settings
            current_max_tabs = (
                max_tabs if max_tabs is not None else (
                    max_windows if max_windows is not None else settings.research_tool.max_tabs
                )
            )

            # Check if save_dir is provided, otherwise use in-memory approach
            save_dir_for_this_task = None
            if settings.research_tool.save_dir:
                # If save_dir is provided, construct the full save directory path for this specific task
                save_dir_for_this_task = str(Path(settings.research_tool.save_dir) / task_id)
                logger.info(f"Deep research save directory for this task: {save_dir_for_this_task}")
            else:
                logger.info("No save_dir configured. Deep research will operate in memory-only mode.")

            logger.info(f"Using max_tabs: {current_max_tabs}")

            result_dict = await agent_instance.run(
                topic=research_task,
                save_dir=save_dir_for_this_task, # Can be None now
                task_id=task_id, # Pass the generated task_id
                max_tabs=current_max_tabs
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


        except Exception as e:
            logger.error(f"Error in run_deep_research: {e}\n{traceback.format_exc()}")
            report_content = f"Error: {e}"

        return report_content

    @server.tool()
    async def run_task(
        ctx: Context,
        task: str,
        max_tabs: Optional[int] = None,
        max_windows: Optional[int] = None,  # deprecated alias
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
            return await run_deep_research(ctx, task, max_tabs=max_tabs, max_windows=max_windows)  # type: ignore

        # auto mode
        if needs_deep_research(task):
            logger.info("Router: escalating to deep research based on heuristic.")
            return await run_deep_research(ctx, task, max_tabs=max_tabs, max_windows=max_windows)  # type: ignore
        else:
            logger.info("Router: using lightweight task (browser agent).")
            return await run_browser_agent(ctx, task)  # type: ignore

    @server.tool()
    async def run_research(
        ctx: Context,
        topic_or_task: str,
        mode: Literal["auto", "task", "research", "deep_research"] = "auto",
        max_tabs: Optional[int] = None,
        max_windows: Optional[int] = None,  # deprecated alias
    ) -> str:
        """
        Unified entry point for browser work with modes:
        - auto (default): choose between task, research (lightweight), and deep_research
        - task: concrete UI/browser actions (e.g., Cloudflare DNS, consoles)
        - research: lightweight research/summarization using a single browser agent
        - deep_research: full deep research pipeline

        Env override: MCP_RESEARCH_MODE=auto|task|research|deep_research
        """
        env_mode = os.getenv("MCP_RESEARCH_MODE", "auto").lower()
        # If caller explicitly picks a mode, honor it; if left at default 'auto', allow env to set default mode
        effective_mode = env_mode if mode == "auto" and env_mode in {"auto", "task", "research", "deep_research"} else mode
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

        # Honor explicit modes first
        if effective_mode in {"task", "research", "deep_research"}:
            chosen_mode = effective_mode
        else:
            # auto routing
            if needs_deep_research(text):
                chosen_mode = "deep_research"
            elif looks_like_task(text):
                chosen_mode = "task"
            elif looks_like_research(text):
                chosen_mode = "research"
            else:
                # default to lightweight task to err on the side of minimal resource usage
                chosen_mode = "task"

        logger.info(f"run_research routing to: {chosen_mode}")

        if chosen_mode == "deep_research":
            return await run_deep_research(ctx, text, max_tabs=max_tabs, max_windows=max_windows)  # type: ignore

        # Lightweight path uses run_browser_agent with a small, mode-specific prefix
        if chosen_mode == "task":
            prefix = (
                "Mode: TASK. Perform the concrete browser actions requested (navigation, forms, toggles, settings). "
                "Favor a single window/tab; act directly and stop when the task is done.\n\n"
            )
        else:  # research
            prefix = (
                "Mode: RESEARCH. Find information, read carefully, and summarize the key findings with links to sources. "
                "Prefer minimal tabs; do not change account settings or perform account logins.\n\n"
            )

        return await run_browser_agent(ctx, f"{prefix}{text}")  # type: ignore

    @server.tool()
    async def run_auto(
        ctx: Context,
        topic_or_task: str,
        max_tabs: Optional[int] = None,
        max_windows: Optional[int] = None,  # deprecated alias
    ) -> str:
        """
        Auto-select between task, research, and deep_research based on heuristics.
        This is equivalent to calling run_research(..., mode="auto").
        """
        return await run_research(ctx, topic_or_task, mode="auto", max_tabs=max_tabs, max_windows=max_windows)  # type: ignore

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
            "  - run_auto(topic_or_task, max_tabs?)\n"
            "  - run_research(topic_or_task, mode=auto|task|research|deep_research, max_tabs?)\n"
            "  - run_task(task, max_tabs?)\n"
            "  - run_deep_research(research_task, max_tabs?)\n"
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

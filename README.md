<img src="./assets/header.png" alt="Better MCP Browser-Use" width="full"/>

<br/>

# Better-MCP-Browser-Use

[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

AI-driven browser automation and web research server for the Model Context Protocol (MCP), with practical upgrades for search control, resource usage, and a simpler one-tool interface.

This is an upgraded fork of the original project:

- Original: https://github.com/Saik0s/mcp-browser-use
- This fork: https://github.com/caarson/mcp-browser-use

1.  **`run_auto`**
  *   **Description:** Auto-selects between `task`, `research`, and `deep_research` using the same heuristics as `run_research(mode="auto")`.
  *   **Arguments:**
    *   `topic_or_task` (string, required)
  *   `max_tabs` (integer, optional)
  *   **Returns:** (string) Final result or research report string.

2.  **`run_research`**

## Why “Better”?

Everything you expect from the original—plus pragmatic upgrades:

- Unified tool: `run_research` with modes `auto | task | research | deep_research` (auto chooses the lightest viable path).
- Smarter resource use: single-window model; deep research runs tabs-only and defaults to 1 concurrent tab (configurable).
- Search engine control: default Bing; supports DuckDuckGo, Brave, Google, or a custom engine template.
1.  **`run_auto`**
  *   **Description:** Auto-selects between `task`, `research`, and `deep_research` using the same heuristics as `run_research(mode="auto")`.
  *   **Arguments:**
    *   `topic_or_task` (string, required): The prompt describing the task or research topic.
    *   `max_tabs` (integer, optional): Upper bound on how many tabs can be used in parallel within a single window (applies when deep research is selected).
  *   **Returns:** (string) Final result or research report string.
### The Essentials
2.  **`run_research`**
     `uvx --from mcp-server-browser-use@latest python -m playwright install`

### Integration Patterns

  *   `max_tabs` (integer, optional): Upper bound on concurrent tabs in a single window (only applied when deep research is selected).

```json
// Example 1A: Use GitHub fork (preferred while PR pending)
"mcpServers": {
  "browser-use": {
    "command": "uvx",
    "args": ["--from", "git+https://github.com/caarson/mcp-browser-use", "mcp-server-browser-use"],
    "env": {
      "MCP_LLM_GOOGLE_API_KEY": "YOUR_KEY_HERE_IF_USING_GOOGLE",
      "MCP_LLM_PROVIDER": "google",
      "MCP_LLM_MODEL_NAME": "gemini-2.5-flash-preview-04-17",
      "MCP_BROWSER_HEADLESS": "true"
    }
  }
}
```

```json
// Example 1B: One-Line Latest Version (PyPI)
"mcpServers": {
    "browser-use": {
      "command": "uvx",
      "args": ["mcp-server-browser-use@latest"],
    }
}
```


```json

> Note: The standalone `run_browser_agent` MCP tool has been removed. Use `run_research` with `mode=task` (for UI actions) or `mode=research` (for lightweight reading/summarization). The heavy pipeline remains available as `run_deep_research`.
// Example 2: Advanced Configuration with CDP
"mcpServers": {
    "browser-use": {
      "command": "uvx",
      "args": ["mcp-server-browser-use@latest"],
      "env": {
        "MCP_LLM_OPENROUTER_API_KEY": "YOUR_KEY_HERE_IF_USING_OPENROUTER",
        "MCP_LLM_PROVIDER": "openrouter",
        "MCP_LLM_MODEL_NAME": "anthropic/claude-3.5-haiku",
        "MCP_LLM_TEMPERATURE": "0.4",

        "MCP_BROWSER_HEADLESS": "false",
        "MCP_BROWSER_WINDOW_WIDTH": "1440",
        "MCP_BROWSER_WINDOW_HEIGHT": "1080",
        "MCP_AGENT_TOOL_USE_VISION": "true",

        "MCP_RESEARCH_TOOL_SAVE_DIR": "/path/to/your/research",
  "MCP_RESEARCH_TOOL_MAX_TABS": "5",

        "MCP_PATHS_DOWNLOADS": "/path/to/your/downloads",

        "MCP_BROWSER_USE_OWN_BROWSER": "true",
        "MCP_BROWSER_CDP_URL": "http://localhost:9222",

        "MCP_AGENT_TOOL_HISTORY_PATH": "/path/to/your/history",

        "MCP_SERVER_LOGGING_LEVEL": "DEBUG",
        "MCP_SERVER_LOG_FILE": "/path/to/your/log/mcp_server_browser_use.log",
      }
    }
}
```

```json
// Example 3: Advanced Configuration with User Data and custom chrome path
"mcpServers": {
    "browser-use": {
      "command": "uvx",
      "args": ["mcp-server-browser-use@latest"],
      "env": {
        "MCP_LLM_OPENAI_API_KEY": "YOUR_KEY_HERE_IF_USING_OPENAI",
        "MCP_LLM_PROVIDER": "openai",
        "MCP_LLM_MODEL_NAME": "gpt-4.1-mini",
        "MCP_LLM_TEMPERATURE": "0.2",

        "MCP_BROWSER_HEADLESS": "false",

        "MCP_BROWSER_BINARY_PATH": "/path/to/your/chrome/binary",
        "MCP_BROWSER_USER_DATA_DIR": "/path/to/your/user/data",
        "MCP_BROWSER_DISABLE_SECURITY": "true",
        "MCP_BROWSER_KEEP_OPEN": "true",
        "MCP_BROWSER_TRACE_PATH": "/path/to/your/trace",

        "MCP_AGENT_TOOL_HISTORY_PATH": "/path/to/your/history",

        "MCP_SERVER_LOGGING_LEVEL": "DEBUG",
        "MCP_SERVER_LOG_FILE": "/path/to/your/log/mcp_server_browser_use.log",
      }
    }
}
```

```json
// Example 4: Local Development Flow (clone and run locally)
"mcpServers": {
    "browser-use": {
      "command": "uv",
      "args": [
        "--directory",
        "/your/dev/path",
        "run",
        "mcp-server-browser-use"
      ],
      "env": {
        "MCP_LLM_OPENROUTER_API_KEY": "YOUR_KEY_HERE_IF_USING_OPENROUTER",
      > Note: The standalone `run_browser_agent` MCP tool has been removed. Use `run_research` with `mode=auto` (default), `mode=task` (UI actions), or `mode=research` (lightweight reading/summarization). The heavy pipeline remains available as `run_deep_research`.
        "MCP_LLM_PROVIDER": "openrouter",
        "MCP_LLM_MODEL_NAME": "openai/gpt-4o-mini",
        "MCP_BROWSER_HEADLESS": "true",
      }
    }
}
```

**Tip:** Start simple (Example 1). Use `.env` to opt into specific features later.

## MCP Tools

This server exposes the following tools via the Model Context Protocol:

### Synchronous Tools (Wait for Completion)

1.  **`run_auto`**
  *   **Description:** Auto-selects between `task`, `research`, and `deep_research` using routing heuristics.
  *   **Arguments:**
    *   `topic_or_task` (string, required): The prompt describing the task or research topic.
  *   `max_tabs` (integer, optional): Upper bound on concurrent tabs in a single window if deep research is chosen.
  *   **Returns:** (string) Final result or research report string.

2.  **`run_research`**
  *   **Description:** Unified entrypoint with modes to handle both browsing tasks and research.
  *   **Arguments:**
    *   `topic_or_task` (string, required): The prompt describing the task or research topic.
    *   `mode` (string, optional): `auto` (default), `task`, `research`, or `deep_research`.
  *   `max_tabs` (integer, optional): Only used when `deep_research` is selected.
  *   **Returns:** (string) Final result or research report string.
  *   **Env override:** `MCP_RESEARCH_MODE=auto|task|research|deep_research` (default: `auto`).

3.  **`run_task`**
  *   **Description:** Smart router that prefers the lightweight task flow and only escalates to deep research when needed. Controlled by `MCP_TASK_ROUTER_MODE`.
  *   **Arguments:**
    *   `task` (string, required): The primary task or objective.
  *   `max_tabs` (integer, optional): Passed through if deep research is chosen to cap concurrent tabs.
  *   **Returns:** (string) The final result or deep research report string.

4.  **`run_deep_research`**
    *   **Description:** Performs in-depth web research on a topic, generates a report, and waits for completion. Uses settings from `MCP_RESEARCH_TOOL_*`, `MCP_LLM_*`, and `MCP_BROWSER_*` environment variables. If `MCP_RESEARCH_TOOL_SAVE_DIR` is set, outputs are saved to a subdirectory within it; otherwise, operates in memory-only mode.
    *   **Arguments:**
    *   `research_task` (string, required): The topic or question for the research.
  *   `max_tabs` (integer, optional): Overrides `MCP_RESEARCH_TOOL_MAX_TABS` from environment (caps simultaneous tabs used by sub-agents). Deprecated aliases supported: `max_windows` and `max_parallel_browsers`.
    *   **Returns:** (string) The generated research report in Markdown format, including the file path (if saved), or an error message.

## CLI Usage

This package also provides a command-line interface `mcp-browser-cli` for direct testing and scripting.

**Global Options:**
*   `--env-file PATH, -e PATH`: Path to a `.env` file to load configurations from.
*   `--log-level LEVEL, -l LEVEL`: Override the logging level (e.g., `DEBUG`, `INFO`).

**Commands:**

1.  **`mcp-browser-cli run-browser-agent [OPTIONS] TASK`**
    *   **Description:** Runs a browser agent task.
    *   **Arguments:**
        *   `TASK` (string, required): The primary task for the agent.
    *   **Example:**
        ```bash
  mcp-browser-cli run-browser-agent "Go to example.com and find the title." -e .env
        ```

2.  **`mcp-browser-cli run-deep-research [OPTIONS] RESEARCH_TASK`**
    *   **Description:** Performs deep web research.
    *   **Arguments:**
        *   `RESEARCH_TASK` (string, required): The topic or question for research.
    *   **Options:**
  *   `--max-tabs INTEGER, -t INTEGER`: Override `MCP_RESEARCH_TOOL_MAX_TABS`. Deprecated aliases supported: `--max-windows/-w`, `--max-parallel-browsers/-p`.
    *   **Example:**
        ```bash
  mcp-browser-cli run-deep-research "What are the latest advancements in AI-driven browser automation?" --max-windows 5 -e .env
        ```

All other configurations (LLM keys, paths, browser settings) are picked up from environment variables (or the specified `.env` file) as detailed in the Configuration section.

### Router behavior (run_task)

`run_task` chooses between a lightweight internal browser agent and `run_deep_research` (heavy). It favors single-window direct completion for tasks like opening GitHub, documentation, or a single README. It escalates only when the prompt clearly indicates multi-source synthesis, investigation/troubleshooting, or when the agent hits blockers.

Control routing with:

```dotenv
# Router mode: auto | always-task | always-research
MCP_TASK_ROUTER_MODE=auto
```

In auto mode, the router uses simple heuristics (keywords for simple navigation vs. complex analysis) and prompt length to decide.

### Unified behavior (run_research)

`run_research` adds an explicit `mode` control:

- `auto` (default): decide between `task`, `research`, and `deep_research`.
- `task`: treat as a concrete UI workflow (e.g., Cloudflare DNS, dashboards, consoles). Uses a single-window bias and acts directly.
- `research`: lightweight reading/summarization using the browser agent with source links; avoids logins/settings changes.
- `deep_research`: invokes the heavier multi-source pipeline.

Env override for default: `MCP_RESEARCH_MODE=auto|task|research|deep_research`.

## Configuration (Environment Variables)

You can configure everything via env vars or a `.env` file. Highlights of the most relevant settings:

- LLM: `MCP_LLM_PROVIDER`, `MCP_LLM_MODEL_NAME`, provider-specific API keys.
- Browser: `MCP_BROWSER_HEADLESS`, `MCP_BROWSER_USE_OWN_BROWSER`, `MCP_BROWSER_CDP_URL`, `MCP_BROWSER_KEEP_OPEN`.
- Agent: `MCP_AGENT_TOOL_MAX_STEPS`, `MCP_AGENT_TOOL_USE_VISION`, `MCP_AGENT_TOOL_HISTORY_PATH`.
- Deep Research: `MCP_RESEARCH_TOOL_SAVE_DIR`, `MCP_RESEARCH_TOOL_MAX_TABS` (default: 1 here). Legacy aliases supported: `MCP_RESEARCH_TOOL_MAX_WINDOWS`, `MCP_RESEARCH_TOOL_MAX_PARALLEL_BROWSERS`.
- Router: `MCP_TASK_ROUTER_MODE=auto|always-task|always-research`.
- Unified mode: `MCP_RESEARCH_MODE=auto|task|research|deep_research`.

### Search engine options

Control search behavior and redirects when the agent initiates searches or navigates to google.* URLs:

```dotenv
# Default engine is Bing
MCP_SEARCH_ENGINE=bing  # options: bing | ddg | google | brave | custom

# Optionally block Google; if true and engine=google, auto-fallback to ddg
MCP_BLOCK_GOOGLE=false

# Example forcing Bing and blocking Google redirects
# MCP_SEARCH_ENGINE=bing
# MCP_BLOCK_GOOGLE=true

# Brave Search (built-in)
# MCP_SEARCH_ENGINE=brave

# Custom search (two options):
# A) URL template with {q}
# MCP_SEARCH_ENGINE=custom
# MCP_SEARCH_URL_TEMPLATE=https://kagi.com/search?q={q}

# B) Base URL + query param name
# MCP_SEARCH_ENGINE=custom
# MCP_SEARCH_ENGINE_URL=https://search.brave.com/search
# MCP_SEARCH_QUERY_PARAM=q
```

## Configuration (Environment Variables)

Configure the server and CLI using environment variables. You can set these in your system or place them in a `.env` file in the project root (use `--env-file` for CLI). Variables are structured with prefixes.

| Variable Group (Prefix)             | Example Variable                               | Description                                                                                                | Default Value                     |
| :---------------------------------- | :--------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :-------------------------------- |
| **Main LLM (MCP_LLM_)**             |                                                | Settings for the primary LLM used by agents.                                                               |                                   |
|                                     | `MCP_LLM_PROVIDER`                             | LLM provider. Options: `openai`, `azure_openai`, `anthropic`, `google`, `mistral`, `ollama`, etc.         | `openai`                          |
|                                     | `MCP_LLM_MODEL_NAME`                           | Specific model name for the provider.                                                                      | `gpt-4.1`                         |
|                                     | `MCP_LLM_TEMPERATURE`                          | LLM temperature (0.0-2.0).                                                                                 | `0.0`                             |
|                                     | `MCP_LLM_BASE_URL`                             | Optional: Generic override for LLM provider's base URL.                                                    | Provider-specific                 |
|                                     | `MCP_LLM_API_KEY`                              | Optional: Generic LLM API key (takes precedence).                                                          | -                                 |
|                                     | `MCP_LLM_OPENAI_API_KEY`                       | API Key for OpenAI (if provider is `openai`).                                                              | -                                 |
|                                     | `MCP_LLM_ANTHROPIC_API_KEY`                    | API Key for Anthropic.                                                                                     | -                                 |
|                                     | `MCP_LLM_GOOGLE_API_KEY`                       | API Key for Google AI (Gemini).                                                                            | -                                 |
|                                     | `MCP_LLM_AZURE_OPENAI_API_KEY`                 | API Key for Azure OpenAI.                                                                                  | -                                 |
|                                     | `MCP_LLM_AZURE_OPENAI_ENDPOINT`                | **Required if using Azure.** Your Azure resource endpoint.                                                 | -                                 |
|                                     | `MCP_LLM_OLLAMA_ENDPOINT`                      | Ollama API endpoint URL.                                                                                   | `http://localhost:11434`          |
|                                     | `MCP_LLM_OLLAMA_NUM_CTX`                       | Context window size for Ollama models.                                                                     | `32000`                           |
| **Planner LLM (MCP_LLM_PLANNER_)**  |                                                | Optional: Settings for a separate LLM for agent planning. Defaults to Main LLM if not set.                |                                   |
|                                     | `MCP_LLM_PLANNER_PROVIDER`                     | Planner LLM provider.                                                                                      | Main LLM Provider                 |
|                                     | `MCP_LLM_PLANNER_MODEL_NAME`                   | Planner LLM model name.                                                                                    | Main LLM Model                    |
| **Browser (MCP_BROWSER_)**          |                                                | General browser settings.                                                                                  |                                   |
|                                     | `MCP_BROWSER_HEADLESS`                         | Run browser without UI (general setting).                                                                  | `false`                           |
|                                     | `MCP_BROWSER_DISABLE_SECURITY`                 | Disable browser security features (general setting, use cautiously).                                       | `false`                           |
|                                     | `MCP_BROWSER_BINARY_PATH`                      | Path to Chrome/Chromium executable.                                                                        | -                                 |
|                                     | `MCP_BROWSER_USER_DATA_DIR`                    | Path to Chrome user data directory.                                                                        | -                                 |
|                                     | `MCP_BROWSER_WINDOW_WIDTH`                     | Browser window width (pixels).                                                                             | `1280`                            |
|                                     | `MCP_BROWSER_WINDOW_HEIGHT`                    | Browser window height (pixels).                                                                            | `1080`                            |
|                                     | `MCP_BROWSER_USE_OWN_BROWSER`                  | Connect to user's browser via CDP URL.                                                                     | `false`                           |
|                                     | `MCP_BROWSER_CDP_URL`                          | CDP URL (e.g., `http://localhost:9222`). Required if `MCP_BROWSER_USE_OWN_BROWSER=true`.                  | -                                 |
|                                     | `MCP_BROWSER_KEEP_OPEN`                        | Keep server-managed browser open between MCP calls (if `MCP_BROWSER_USE_OWN_BROWSER=false`).               | `false`                           |
|                                     | `MCP_BROWSER_TRACE_PATH`                       | Optional: Directory to save Playwright trace files. If not set, tracing to file is disabled.               | ` ` (empty, tracing disabled)     |
| **Agent Tool (MCP_AGENT_TOOL_)**    |                                                | Settings for the `run_browser_agent` tool.                                                                 |                                   |
|                                     | `MCP_AGENT_TOOL_MAX_STEPS`                     | Max steps per agent run.                                                                                   | `100`                             |
|                                     | `MCP_AGENT_TOOL_MAX_ACTIONS_PER_STEP`          | Max actions per agent step.                                                                                | `5`                               |
|                                     | `MCP_AGENT_TOOL_TOOL_CALLING_METHOD`           | Method for tool invocation ('auto', 'json_schema', 'function_calling').                                    | `auto`                            |
|                                     | `MCP_AGENT_TOOL_MAX_INPUT_TOKENS`              | Max input tokens for LLM context.                                                                          | `128000`                          |
|                                     | `MCP_AGENT_TOOL_USE_VISION`                    | Enable vision capabilities (screenshot analysis).                                                          | `true`                            |
|                                     | `MCP_AGENT_TOOL_HEADLESS`                      | Override `MCP_BROWSER_HEADLESS` for this tool (true/false/empty).                                          | ` ` (uses general)                |
|                                     | `MCP_AGENT_TOOL_DISABLE_SECURITY`              | Override `MCP_BROWSER_DISABLE_SECURITY` for this tool (true/false/empty).                                  | ` ` (uses general)                |
|                                     | `MCP_AGENT_TOOL_ENABLE_RECORDING`              | Enable Playwright video recording.                                                                         | `false`                           |
|                                     | `MCP_AGENT_TOOL_SAVE_RECORDING_PATH`           | Optional: Path to save recordings. If not set, recording to file is disabled even if `ENABLE_RECORDING=true`. | ` ` (empty, recording disabled)   |
|                                     | `MCP_AGENT_TOOL_HISTORY_PATH`                  | Optional: Directory to save agent history JSON files. If not set, history saving is disabled.              | ` ` (empty, history saving disabled) |
| **Research Tool (MCP_RESEARCH_TOOL_)** |                                             | Settings for the `run_deep_research` tool.                                                                 |                                   |
|                                     | `MCP_RESEARCH_TOOL_MAX_TABS`                   | Max concurrent tabs within a single browser window for deep research. Deprecated aliases: `MCP_RESEARCH_TOOL_MAX_WINDOWS`, `MCP_RESEARCH_TOOL_MAX_PARALLEL_BROWSERS`. | `1` |
|                                     | `MCP_RESEARCH_TOOL_SAVE_DIR`                   | Optional: Base directory to save research artifacts. Task ID will be appended. If not set, operates in memory-only mode. | `None`                           |
| **Paths (MCP_PATHS_)**              |                                                | General path settings.                                                                                     |                                   |
|                                     | `MCP_PATHS_DOWNLOADS`                          | Optional: Directory for downloaded files. If not set, persistent downloads to a specific path are disabled.  | ` ` (empty, downloads disabled)  |
| **Server (MCP_SERVER_)**            |                                                | Server-specific settings.                                                                                  |                                   |
|                                     | `MCP_SERVER_LOG_FILE`                          | Path for the server log file. Empty for stdout.                                                            | ` ` (empty, logs to stdout)       |
|                                     | `MCP_SERVER_LOGGING_LEVEL`                     | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).                                           | `ERROR`                           |
|                                     | `MCP_SERVER_ANONYMIZED_TELEMETRY`              | Enable/disable anonymized telemetry (`true`/`false`).                                                      | `true`                            |
|                                     | `MCP_SERVER_MCP_CONFIG`                        | Optional: JSON string for MCP client config used by the internal controller.                               | `null`                            |

| **Search (MCP_*)**                  |                                                | Web search engine configuration for the built-in "search_google" action.                                   |                                   |
|                                     | `MCP_SEARCH_ENGINE`                            | Which search engine to use when the agent triggers a search. Options: `bing`, `ddg`, `google`, `brave`, `custom`. | `bing`                            |
|                                     | `MCP_BLOCK_GOOGLE`                             | If `true` and the engine is `google`, the search will automatically fall back to DuckDuckGo. Also rewrites Google search URLs before navigation. | `false`                           |

**Supported LLM Providers (`MCP_LLM_PROVIDER`):**
`openai`, `azure_openai`, `anthropic`, `google`, `mistral`, `ollama`, `deepseek`, `openrouter`, `alibaba`, `moonshot`, `unbound`

*(Refer to `.env.example` for a comprehensive list of all supported environment variables and their specific provider keys/endpoints.)*

### Search engine options

Set these to control the built-in search action behavior:

```dotenv
# Default engine is Bing
MCP_SEARCH_ENGINE=bing  # options: bing | ddg | google | brave | custom

# Optionally block Google; if true and engine=google, auto-fallback to ddg
MCP_BLOCK_GOOGLE=false

# Example forcing Bing and blocking Google redirects
# MCP_SEARCH_ENGINE=bing
# MCP_BLOCK_GOOGLE=true

# Example using Brave Search (built-in)
# MCP_SEARCH_ENGINE=brave

# Use a custom engine (two options):
# A) Provide a URL template that includes {q}
# MCP_SEARCH_ENGINE=custom
# MCP_SEARCH_URL_TEMPLATE=https://kagi.com/search?q={q}

# B) Provide a base URL and the query param name
# MCP_SEARCH_ENGINE=custom
# MCP_SEARCH_ENGINE_URL=https://search.brave.com/search
# MCP_SEARCH_QUERY_PARAM=q
```

## Connecting to Your Own Browser (CDP)

Instead of having the server launch and manage its own browser instance, you can connect it to a Chrome/Chromium browser that you launch and manage yourself.

**Steps:**

1.  **Launch Chrome/Chromium with Remote Debugging Enabled:**
    (Commands for macOS, Linux, Windows as previously listed, e.g., `google-chrome --remote-debugging-port=9222`)

2.  **Configure Environment Variables:**
    Set the following environment variables:
    ```dotenv
    MCP_BROWSER_USE_OWN_BROWSER=true
    MCP_BROWSER_CDP_URL=http://localhost:9222 # Use the same port
    # Optional: MCP_BROWSER_USER_DATA_DIR=/path/to/your/profile
    ```

3.  **Run the MCP Server or CLI:**
    Start the server (`uv run mcp-server-browser-use`) or CLI (`mcp-browser-cli ...`) as usual.

**Important Considerations:**
*   The browser launched with `--remote-debugging-port` must remain open.
*   Settings like `MCP_BROWSER_HEADLESS` and `MCP_BROWSER_KEEP_OPEN` are ignored when `MCP_BROWSER_USE_OWN_BROWSER=true`.

## Development

```bash
# Install dev dependencies and sync project deps
uv sync --dev

# Install playwright browsers
uv run playwright install

# Run MCP server with debugger (Example connecting to own browser via CDP)
# 1. Launch Chrome: google-chrome --remote-debugging-port=9222 --user-data-dir="optional/path/to/user/profile"
# 2. Run inspector command with environment variables:
npx @modelcontextprotocol/inspector@latest \
  -e MCP_LLM_GOOGLE_API_KEY=$GOOGLE_API_KEY \
  -e MCP_LLM_PROVIDER=google \
  -e MCP_LLM_MODEL_NAME=gemini-2.5-flash-preview-04-17 \
  -e MCP_BROWSER_USE_OWN_BROWSER=true \
  -e MCP_BROWSER_CDP_URL=http://localhost:9222 \
  -e MCP_RESEARCH_TOOL_SAVE_DIR=./tmp/dev_research_output \
  uv --directory . run mcp-server-browser-use

# Note: Change timeout in inspector's config panel if needed (default is 10 seconds)

# Run CLI example
# Create a .env file with your settings (including MCP_RESEARCH_TOOL_SAVE_DIR) or use environment variables
uv run mcp-browser-cli -e .env run-browser-agent "What is the title of example.com?"
uv run mcp-browser-cli -e .env run-deep-research "What is the best material for a pan for everyday use on amateur kitchen and dishwasher?"

# Prefer running from the GitHub fork while PR is pending
uvx --from git+https://github.com/caarson/mcp-browser-use mcp-server-browser-use
```

## Troubleshooting

-   **Configuration Error on Startup**: If the application fails to start with an error about a missing setting, ensure all **mandatory** environment variables (like `MCP_RESEARCH_TOOL_SAVE_DIR`) are set correctly in your environment or `.env` file.
-   **Browser Conflicts**: If *not* using CDP (`MCP_BROWSER_USE_OWN_BROWSER=false`), ensure no conflicting Chrome instances are running with the same user data directory if `MCP_BROWSER_USER_DATA_DIR` is specified.
-   **CDP Connection Issues**: If using `MCP_BROWSER_USE_OWN_BROWSER=true`:
    *   Verify Chrome was launched with `--remote-debugging-port`.
    *   Ensure the port in `MCP_BROWSER_CDP_URL` matches.
    *   Check firewalls and ensure the browser is running.
-   **API Errors**: Double-check API keys (`MCP_LLM_<PROVIDER>_API_KEY` or `MCP_LLM_API_KEY`) and endpoints (e.g., `MCP_LLM_AZURE_OPENAI_ENDPOINT` for Azure).
-   **Vision Issues**: Ensure `MCP_AGENT_TOOL_USE_VISION=true` and your LLM supports vision.
-   **Dependency Problems**: Run `uv sync` and `uv run playwright install`.
-   **File/Path Issues**:
    *   If optional features like history saving, tracing, or downloads are not working, ensure the corresponding path variables (`MCP_AGENT_TOOL_HISTORY_PATH`, `MCP_BROWSER_TRACE_PATH`, `MCP_PATHS_DOWNLOADS`) are set and the application has write permissions to those locations.
    *   For deep research, ensure `MCP_RESEARCH_TOOL_SAVE_DIR` is set to a valid, writable directory.
-   **Logging**: Check the log file (`MCP_SERVER_LOG_FILE`, if set) or console output. Increase `MCP_SERVER_LOGGING_LEVEL` to `DEBUG` for more details. For CLI, use `--log-level DEBUG`.

## License

MIT - See [LICENSE](LICENSE) for details.

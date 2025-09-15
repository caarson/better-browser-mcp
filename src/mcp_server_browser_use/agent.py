"""Public agent API for mcp_server_browser_use.

Prefer importing TaskAgent and ResearchAgent from here instead of _internal paths.
"""
from ._internal.agent import TaskAgent, ResearchAgent

__all__ = ["TaskAgent", "ResearchAgent"]

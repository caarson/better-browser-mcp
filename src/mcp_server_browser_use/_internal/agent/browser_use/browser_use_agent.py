"""Compatibility shim for the legacy BrowserUseAgent path.

The implementation has moved to `_internal.agent.task_agent_impl.BrowserUseAgent`.
Keep this file importing and re-exporting for backward compatibility.
"""

from ..task_agent_impl import BrowserUseAgent  # type: ignore F401

__all__ = ["BrowserUseAgent"]
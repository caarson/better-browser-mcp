"""Compatibility shim for the legacy DeepResearchAgent path.

The implementation is now centralized under `_internal.agent.research_agent_impl`.
Keep this file as a re-export to avoid breaking imports.
"""

from ..research_agent_impl import *  # noqa: F401,F403

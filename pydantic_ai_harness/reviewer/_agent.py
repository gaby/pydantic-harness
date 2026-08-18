"""Runnable agent instance for the `Reviewer` harness."""

from pydantic_ai import Agent

from pydantic_ai_harness.reviewer._capability import Reviewer

reviewer_agent = Agent(name='reviewer', capabilities=[Reviewer()])
"""Model-less code-review agent for CLIs that load `module:variable` targets."""

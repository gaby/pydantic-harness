"""Complete code-review harness assembled from regular capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_REVIEW_COMMANDS: tuple[str, ...] = (
    'git',
    'rg',
    'grep',
    'find',
    'ls',
    'cat',
    'head',
    'tail',
    'wc',
    'diff',
)
"""Commands available to `Reviewer` unless an explicit allowlist is supplied.

Read-oriented: the review reaches the change through `git diff` and `git log`
and searches the tree from there. No build, test, or edit commands are included,
so a review that should run the test suite needs its own allowlist.
"""

DEFAULT_REVIEWER_INSTRUCTIONS = """\
Review the change against the repository's own conventions before any general style preference.
Read the code around a change before judging whether it is correct.
Report each finding with its file and line, what breaks, and the input or state that triggers it.
Separate what you confirmed in the code from what you suspect.
Leave working code alone unless a finding names a concrete defect or a violated convention.
Say so plainly when a change needs nothing.
"""
"""Default instructions for `Reviewer`."""


def _inspector(workspace: str | Path, allowed_commands: Sequence[str]) -> SubAgent[AgentDepsT]:
    agent = Agent[AgentDepsT](  # pyright: ignore[reportCallIssue, reportArgumentType]
        name='inspector',
        description='Review one file or area of the change and report findings with file and line references',
        instructions='Report findings with concrete paths, line numbers, and the evidence you read.',
        capabilities=[
            FileSystem[AgentDepsT](workspace, read_only=True),
            Shell[AgentDepsT](
                cwd=workspace,
                allowed_commands=allowed_commands,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            RepoContext[AgentDepsT](workspace_dir=Path(workspace)),
            ToolOutputLimits[AgentDepsT](),
        ],
    )
    return SubAgent(agent)


class Reviewer(CombinedCapability[AgentDepsT]):
    """A complete code-review harness built as a regular combined capability.

    See the class definition and [Reviewer docs](https://pydantic.dev/docs/ai/harness/reviewer/) for the exact
    composition.

    The agent reads the workspace and runs read-oriented commands; it gets no file write, edit, or delete tools.
    That is a guardrail against accidental edits, not a write boundary: command validation checks only the first
    token, so `git` reaches subcommands that write. Review untrusted branches inside an OS-level sandbox such as
    `ModalSandbox` or a container.

    It comes with concise default instructions. Pass `instructions=` to replace them, or `instructions=None` to run
    with no default instructions.
    """

    def __init__(
        self,
        workspace: str | Path = '.',
        *,
        allowed_commands: Sequence[str] | None = None,
        subagents: Sequence[SubAgent[AgentDepsT]] | None = None,
        instructions: str | None = DEFAULT_REVIEWER_INSTRUCTIONS,
    ) -> None:
        commands = DEFAULT_REVIEW_COMMANDS if allowed_commands is None else allowed_commands
        delegates = [_inspector(workspace, commands)] if subagents is None else subagents
        capabilities: list[AbstractCapability[AgentDepsT]] = []
        if instructions is not None:
            capabilities.append(Capability[AgentDepsT](instructions=instructions))
        capabilities.extend(
            [
                FileSystem[AgentDepsT](workspace, read_only=True),
                Shell[AgentDepsT](
                    cwd=workspace,
                    allowed_commands=commands,
                    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
                ),
                RepoContext[AgentDepsT](workspace_dir=Path(workspace)),
                Planning[AgentDepsT](),
            ]
        )
        if delegates:
            capabilities.append(SubAgents[AgentDepsT](agents=delegates, agent_folders=None))
        capabilities.extend(
            [
                ClearToolResults[AgentDepsT](max_fraction=0.7),
                WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
                ToolOutputLimits[AgentDepsT](),
            ]
        )
        super().__init__(capabilities)

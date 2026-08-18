"""Complete code-review harness assembled from regular capabilities."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolGuardrail
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

DEFAULT_REVIEWER_COMMANDS: tuple[str, ...] = (
    'git',
    'rg',
    'grep',
    'ls',
    'cat',
    'head',
    'tail',
    'wc',
    'diff',
)
"""Commands available to `Reviewer` unless an explicit allowlist is supplied.

Every entry reads. `find` is absent on purpose: `-delete` and `-exec` make it a
write and execution primitive that first-token validation cannot distinguish
from a search, and `find_files` covers the search. `git` is restricted further
to the read-only subcommands in `READ_ONLY_GIT_SUBCOMMANDS`.

A supplied allowlist replaces this one, and an empty one drops the shell
entirely rather than disabling the check. `Shell` reads an empty allowlist as
"no allowlist", which would hand a review agent an unrestricted shell.
"""

READ_ONLY_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        'blame',
        'cat-file',
        'describe',
        'diff',
        'diff-index',
        'diff-tree',
        'for-each-ref',
        'grep',
        'log',
        'ls-files',
        'ls-remote',
        'ls-tree',
        'merge-base',
        'name-rev',
        'rev-list',
        'rev-parse',
        'shortlog',
        'show',
        'status',
        'symbolic-ref',
        'whatchanged',
    }
)
"""`git` subcommands a review may run.

An allowlist rather than a denylist of writing subcommands: `git` grows
subcommands faster than this set can track them, and an unrecognized one should
be refused rather than run.
"""

DENIED_SHELL_OPERATORS: tuple[str, ...] = ('>', '|', ';', '&', '`', '$(', '\n')
"""Shell metacharacters `Reviewer` refuses in a command.

Commands reach `/bin/sh` as a string while the allowlist only validates the
first token, so without this every allowed command is a write primitive
(`cat a > b`) and an execution primitive (`ls | sh`). Blocking the operators is
what makes the allowlist mean what it says. The cost is that pipes and
alternation (`rg 'a|b'`) are unavailable; run the steps as separate commands, or
supply your own `Shell` from the blown-out form.
"""

SECRET_PATH_PATTERNS: tuple[str, ...] = ('.env', '.env.*', '*.pem', '*.key', '**/secrets*')
"""Workspace paths the review's file tools refuse to read.

`FileSystem.protected_patterns` covers the same paths but gates writes only, so
under `read_only=True` it is inert: without this the reviewer reads every
credential in the tree. The shell is a separate surface (`cat .env` still
works), which is one more reason to review untrusted branches in a sandbox.
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

_INSPECTOR_INSTRUCTIONS = 'Report findings with concrete paths, line numbers, and the evidence you read.'

SHELL_COMMAND_TOOLS: tuple[str, ...] = ('run_command', 'start_command')
"""`Shell` tools that take a command, for scoping the guardrail to them."""

_GIT_SUBCOMMAND_REFUSAL = (
    f'Only read-only `git` subcommands are available: {", ".join(sorted(READ_ONLY_GIT_SUBCOMMANDS))}. '
    'Write `git <subcommand>` with no options before it.'
)

_INSPECTOR_DESCRIPTION = 'Review one file or area of the change and report findings with file and line references'


def review_command_guard(call: ToolCallInfo) -> GuardrailResult:
    """Refuse shell commands the first-token allowlist would let through.

    Two cases it cannot see: a `git` subcommand that writes, since every `git`
    subcommand looks alike to a first-token check, and quoting that `shlex`
    rejects but `/bin/sh` accepts, which `ShellToolset._check_command` treats as
    "nothing to validate" and permits. Metacharacters are refused a layer down,
    by `Shell(denied_operators=DENIED_SHELL_OPERATORS)`.
    """
    command = call.args.get('command')
    if not isinstance(command, str):  # pragma: no cover - the toolset's own schema keeps this a string
        return GuardrailResult.allow()

    try:
        tokens = shlex.split(command)
    except ValueError:
        return GuardrailResult.block('That command is not quoted in a way this agent can check. Rewrite it.')
    if tokens[:1] == ['git'] and (len(tokens) < 2 or tokens[1] not in READ_ONLY_GIT_SUBCOMMANDS):
        return GuardrailResult.block(_GIT_SUBCOMMAND_REFUSAL)
    return GuardrailResult.allow()


def _review_tools(
    workspace: Path, commands: Sequence[str], instructions: str | None
) -> list[AbstractCapability[AgentDepsT]]:
    """The read-only surface both the reviewer and its delegate work through.

    Built once so a delegate can never end up with a wider workspace than the
    agent it reports to.
    """
    capabilities: list[AbstractCapability[AgentDepsT]] = []
    if instructions is not None:
        capabilities.append(Capability[AgentDepsT](instructions=instructions))
    capabilities.append(
        FileSystem[AgentDepsT](workspace, read_only=True, denied_patterns=SECRET_PATH_PATTERNS),
    )
    if commands:
        capabilities.extend(
            [
                Shell[AgentDepsT](
                    cwd=workspace,
                    allowed_commands=commands,
                    denied_operators=DENIED_SHELL_OPERATORS,
                    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
                ),
                ToolGuardrail[AgentDepsT](guard=review_command_guard, tools=SHELL_COMMAND_TOOLS),
            ]
        )
    return capabilities


def _inspector(workspace: Path, commands: Sequence[str], instructions: str | None) -> SubAgent[AgentDepsT]:
    capabilities: list[AbstractCapability[AgentDepsT]] = _review_tools(workspace, commands, instructions)
    capabilities.extend(
        [
            # The parent already carries the workspace instruction files; re-reading them into
            # every delegation duplicates them in an uncached prefix.
            RepoContext[AgentDepsT](workspace_dir=workspace, autoload_instructions=False),
            ClearToolResults[AgentDepsT](max_fraction=0.7),
            WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
            # Truncate rather than the default spill: a delegation's handles die with its run,
            # so a spilled file is unreadable the moment the inspector reports back.
            ToolOutputLimits[AgentDepsT](bands=[Band(over=10_000, action=Truncate())]),
        ]
    )
    agent = Agent[AgentDepsT](  # pyright: ignore[reportCallIssue, reportArgumentType]
        name='inspector',
        description=_INSPECTOR_DESCRIPTION,
        instructions=_INSPECTOR_INSTRUCTIONS,
        capabilities=capabilities,
    )
    return SubAgent(agent)


class Reviewer(CombinedCapability[AgentDepsT]):
    """A complete code-review harness built as a regular combined capability.

    See the class definition and [Reviewer docs](https://pydantic.dev/docs/ai/harness/reviewer/) for the exact
    composition.

    The agent reads the workspace and runs read-oriented commands. It gets no file write or edit tools, secrets
    (`.env`, `*.pem`, `*.key`, `**/secrets*`) are unreadable, shell metacharacters are refused so an allowed command
    cannot redirect or chain into a write, and `git` is limited to read-only subcommands. That is a guardrail against
    a review that edits, not a security boundary: the shell still reads any file the allowlisted commands can open,
    and an allowlisted binary can do more than its name suggests. Review untrusted branches inside an OS-level
    sandbox such as `ModalSandbox` or a container.

    `RepoContext` loads the workspace's `AGENTS.md`/`CLAUDE.md` into the agent's instructions, and the default
    instructions tell the model to review against them. On a branch you do not trust, those files are part of the
    change under review: pass `instructions=` of your own, or drop `RepoContext` from the blown-out form.

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
        root = Path(workspace).expanduser()
        commands = tuple(DEFAULT_REVIEWER_COMMANDS if allowed_commands is None else allowed_commands)
        delegates = [_inspector(root, commands, instructions)] if subagents is None else subagents
        capabilities: list[AbstractCapability[AgentDepsT]] = _review_tools(root, commands, instructions)
        capabilities.extend(
            [
                RepoContext[AgentDepsT](workspace_dir=root),
                Planning[AgentDepsT](),
            ]
        )
        if delegates:
            capabilities.append(SubAgents[AgentDepsT](agents=delegates, agent_folders=None))
        capabilities.extend(
            [
                # The delegates' findings are this agent's deliverable, not scratch work:
                # clearing them would erase the review before it is written.
                ClearToolResults[AgentDepsT](max_fraction=0.7, exclude_tools=frozenset({'delegate_task'})),
                WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
                ToolOutputLimits[AgentDepsT](),
            ]
        )
        super().__init__(capabilities)

"""Build a code-review agent from the blocks packaged as `Reviewer`.

Run the packaged equivalent without assembling the blocks:

    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.reviewer:reviewer_agent -m anthropic:claude-fable-5
"""

import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models import Model

from pydantic_ai_harness import (
    LLM_API_KEY_ENV_PATTERNS,
    ClearToolResults,
    FileSystem,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolGuardrail,
    ToolOutputLimits,
    WarnNearLimits,
)
from pydantic_ai_harness.reviewer import review_command_guard
from pydantic_ai_harness.tool_output_limits import Band, Truncate

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')

# Keep this blown-out composition in sync across docs/reviewer.md,
# pydantic_ai_harness/reviewer/README.md, and examples/review_agent.py.
INSTRUCTIONS = """\
Review the change against the repository's own conventions before any general style preference.
Read the code around a change before judging whether it is correct.
Report each finding with its file and line, what breaks, and the input or state that triggers it.
Separate what you confirmed in the code from what you suspect.
Leave working code alone unless a finding names a concrete defect or a violated convention.
Say so plainly when a change needs nothing.
"""

# Every command reads. `find` is absent on purpose: `-delete` and `-exec` are writes
# that first-token validation cannot tell from a search.
REVIEW_COMMANDS = ('git', 'rg', 'grep', 'ls', 'cat', 'head', 'tail', 'wc', 'diff')
# Without these, any allowed command redirects into a file or pipes into an interpreter.
DENIED_OPERATORS = ('>', '|', ';', '&', '`', '$(', '\n')
# `protected_patterns` gates writes only, so a read-only agent needs these denied.
SECRET_PATTERNS = ('.env', '.env.*', '*.pem', '*.key', '**/secrets*')


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the code-review agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    inspector = SubAgent(
        Agent(
            name='inspector',
            description='Review one file or area of the change and report findings with file and line references',
            instructions='Report findings with concrete paths, line numbers, and the evidence you read.',
            capabilities=[
                Capability(instructions=INSTRUCTIONS),
                FileSystem(workspace, read_only=True, denied_patterns=SECRET_PATTERNS),
                Shell(
                    cwd=workspace,
                    allowed_commands=REVIEW_COMMANDS,
                    denied_operators=DENIED_OPERATORS,
                    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
                ),
                ToolGuardrail(guard=review_command_guard),
                # The parent already carries the workspace instruction files.
                RepoContext(workspace_dir=workspace, autoload_instructions=False),
                ClearToolResults(max_fraction=0.7),
                WarnNearLimits(max_context_fraction=0.9),
                # Truncate rather than spill: a delegation's handles die with its run.
                ToolOutputLimits(bands=[Band(over=10_000, action=Truncate())]),
            ],
        )
    )
    return Agent(
        model,
        name='reviewer',
        capabilities=[
            Capability(instructions=INSTRUCTIONS),  # the default review contract, replaceable per run
            FileSystem(workspace, read_only=True, denied_patterns=SECRET_PATTERNS),  # read and search, no secrets
            Shell(  # read-oriented allowlist, no operators, LLM API keys stripped from the environment
                cwd=workspace,
                allowed_commands=REVIEW_COMMANDS,
                denied_operators=DENIED_OPERATORS,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            ToolGuardrail(guard=review_command_guard),  # read-only `git`; refuses commands it cannot parse
            RepoContext(workspace_dir=workspace),  # loads AGENTS.md/CLAUDE.md + repo structure
            Planning(),  # structured review plans the model maintains
            SubAgents(agents=[inspector], agent_folders=None),  # delegate a file or area off the main context
            ClearToolResults(  # clears old tool results near the limit, except the delegates' findings
                max_fraction=0.7,
                exclude_tools=frozenset({'delegate_task'}),
            ),
            WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
            ToolOutputLimits(),  # bounds oversized tool results
        ],
    )


def main() -> None:
    """Start an interactive review session in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()

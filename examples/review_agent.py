"""Build a code-review agent from the blocks packaged as `Reviewer`.

Run the packaged equivalent without assembling the blocks:

    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.reviewer:reviewer_agent -m anthropic:claude-fable-5
"""

import os
from pathlib import Path

from pydantic_ai import Agent
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
    ToolOutputLimits,
    WarnNearLimits,
)

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')

# Keep this blown-out composition in sync with pydantic_ai_harness/reviewer,
# pydantic_ai_harness/reviewer/README.md, and docs/reviewer.md.
INSTRUCTIONS = """\
Review the change against the repository's own conventions before any general style preference.
Read the code around a change before judging whether it is correct.
Report each finding with its file and line, what breaks, and the input or state that triggers it.
Separate what you confirmed in the code from what you suspect.
Leave working code alone unless a finding names a concrete defect or a violated convention.
Say so plainly when a change needs nothing.
"""

# Read-oriented: the review reaches the change through git and searches from there.
# No build or test commands, so a review that runs the suite adds them here.
REVIEW_COMMANDS = (
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


def build_agent(model: Model | str = DEFAULT_MODEL, workspace: Path | None = None) -> Agent:
    """Build the code-review agent for `workspace` (defaults to the current directory)."""
    workspace = workspace or Path.cwd()
    inspector = SubAgent(
        Agent(
            name='inspector',
            description='Review one file or area of the change and report findings with file and line references',
            instructions='Report findings with concrete paths, line numbers, and the evidence you read.',
            capabilities=[
                FileSystem(workspace, read_only=True),
                Shell(
                    cwd=workspace,
                    allowed_commands=REVIEW_COMMANDS,
                    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
                ),
                RepoContext(workspace_dir=workspace),
                ToolOutputLimits(),
            ],
        )
    )
    return Agent(
        model,
        name='reviewer',
        instructions=INSTRUCTIONS,
        capabilities=[
            FileSystem(workspace, read_only=True),  # read, list, and search only
            Shell(  # read-oriented allowlist, LLM API keys stripped from their environment
                cwd=workspace,
                allowed_commands=REVIEW_COMMANDS,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            RepoContext(workspace_dir=workspace),  # loads AGENTS.md/CLAUDE.md + repo structure
            Planning(),  # structured review plans the model maintains
            SubAgents(agents=[inspector], agent_folders=None),  # delegate a file or area off the main context
            ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
            WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
            ToolOutputLimits(),  # bounds oversized tool results
        ],
    )


def main() -> None:
    """Start an interactive review session in the current repository."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()

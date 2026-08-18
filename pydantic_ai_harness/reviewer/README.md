# Reviewer

`Reviewer` gives a Pydantic AI agent a stack for reviewing a change in a local codebase and reporting findings against the repository's own conventions.
It is a regular [combined capability](https://pydantic.dev/docs/ai/capabilities/custom/#composition-and-middleware-semantics) made from the [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) below, so you can use it as-is or take it apart.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Reviewer

agent = Agent('anthropic:claude-fable-5', capabilities=[Reviewer('.')])

result = agent.run_sync('Review the working tree against origin/main.')
print(result.output)
#> ...
```

It is `Coder` without the write tools: the same workspace, repository context, and delegation, with a read-only filesystem and a read-oriented command allowlist, so the run ends in findings rather than edits.

The same agent works with every Pydantic AI interface: [`agent.to_cli_sync()`](https://pydantic.dev/docs/ai/cli/) for terminal chat, [`agent.to_web()`](https://pydantic.dev/docs/ai/web/) for a browser chat UI.

Or skip the file entirely and run the exported `reviewer_agent` with [`clai`](https://pydantic.dev/docs/ai/cli/#custom-agents) (the Pydantic AI CLI), via [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.reviewer:reviewer_agent -m anthropic:claude-fable-5
```

It is literally these capabilities combined, in this order:

- Concise default review instructions (`DEFAULT_REVIEWER_INSTRUCTIONS`): pass `instructions='...'` to replace them, or `instructions=None` to disable
- [`FileSystem(read_only=True)`](https://pydantic.dev/docs/ai/harness/filesystem/): read, list, and search tools rooted at the workspace, path-traversal and symlink safe; no write, edit, or delete tool is exposed
- [`Shell`](https://pydantic.dev/docs/ai/harness/shell/): a read-oriented allowlist (`DEFAULT_REVIEW_COMMANDS`) rooted at the workspace, with common LLM provider API-key variables filtered from inherited command environments
- [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/): the `AGENTS.md`/`CLAUDE.md` conventions the change is reviewed against, plus repository structure
- [`Planning`](https://pydantic.dev/docs/ai/harness/planning/): a plan the agent keeps current while it works through a large diff
- [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/): delegation, with an `inspector` sub-agent for one file or area by default
- [`ClearToolResults`](https://pydantic.dev/docs/ai/harness/compaction/): clears stale tool results at 70% of the model context window
- [`WarnNearLimits`](https://pydantic.dev/docs/ai/harness/compaction/): warns the agent at 90% of the model context window
- [`ToolOutputLimits`](https://pydantic.dev/docs/ai/harness/tool-output-limits/): bounds how much context any single tool result can consume

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries. Pass `allowed_commands=[...]` to change what the review may run; the list threads through to the `inspector` sub-agent too.

## Findings as data

Give the agent an [`output_type`](https://pydantic.dev/docs/ai/output/) and the review returns structured findings instead of prose:

```python
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai_harness import Reviewer


@dataclass
class Finding:
    file: str
    line: int
    problem: str
    trigger: str


agent = Agent('anthropic:claude-fable-5', output_type=list[Finding], capabilities=[Reviewer('.')])

result = agent.run_sync('Review the working tree against origin/main.')
print(result.output)
#> ...
```

## Read-only, and what that means

Withholding the write tools stops the agent from editing the workspace through its filesystem tools. It is not a write boundary: command validation checks only the first token, so `git` reaches subcommands that write, and the shell can reach any binary the allowlist names. For review of untrusted branches, run the agent inside an OS-level sandbox such as [`ModalSandbox`](https://pydantic.dev/docs/ai/harness/modal-sandbox/) or a container.

`DEFAULT_REVIEW_COMMANDS` contains no build or test commands, so a review that should run the suite needs its own allowlist:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DEFAULT_REVIEW_COMMANDS, Reviewer

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[Reviewer('.', allowed_commands=[*DEFAULT_REVIEW_COMMANDS, 'pytest'])],
)
```

## Blown-out equivalent

<!-- Keep this in sync with pydantic_ai_harness/reviewer, docs/reviewer.md, and examples/review_agent.py; it intentionally shows the complete picture. -->

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import (
    ClearToolResults,
    FileSystem,
    LLM_API_KEY_ENV_PATTERNS,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolOutputLimits,
    WarnNearLimits,
)

instructions = """\
Review the change against the repository's own conventions before any general style preference.
Read the code around a change before judging whether it is correct.
Report each finding with its file and line, what breaks, and the input or state that triggers it.
Separate what you confirmed in the code from what you suspect.
Leave working code alone unless a finding names a concrete defect or a violated convention.
Say so plainly when a change needs nothing.
"""

review_commands = ['git', 'rg', 'grep', 'find', 'ls', 'cat', 'head', 'tail', 'wc', 'diff']

inspector = SubAgent(
    Agent(
        name='inspector',
        description='Review one file or area of the change and report findings with file and line references',
        instructions='Report findings with concrete paths, line numbers, and the evidence you read.',
        capabilities=[
            FileSystem('.', read_only=True),
            Shell(
                cwd='.',
                allowed_commands=review_commands,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            RepoContext(workspace_dir=Path('.')),
            ToolOutputLimits(),
        ],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    name='reviewer',
    instructions=instructions,
    capabilities=[
        FileSystem('.', read_only=True),  # read, list, and search only
        Shell(  # read-oriented allowlist, LLM API keys stripped from their environment
            cwd='.',
            allowed_commands=review_commands,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ),
        RepoContext(workspace_dir=Path('.')),  # loads AGENTS.md/CLAUDE.md + repo structure
        Planning(),  # structured review plans the model maintains
        SubAgents(agents=[inspector], agent_folders=None),  # delegate a file or area off the main context
        ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
        WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
        ToolOutputLimits(),  # bounds oversized tool results
    ],
)
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/reviewer/). While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade; see the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

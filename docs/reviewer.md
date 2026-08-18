---
title: Reviewer
description: A complete Pydantic AI code-review harness that reads a change and reports findings.
---

# Reviewer

`Reviewer` gives a Pydantic AI agent a stack for reviewing a change in a local codebase and reporting findings against the repository's own conventions.
It is a regular [combined capability](https://pydantic.dev/docs/ai/capabilities/custom/#composition-and-middleware-semantics) made from the [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) below, so you can use it as-is or take it apart.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

Point it at a workspace and ask for a review:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Reviewer

agent = Agent('anthropic:claude-fable-5', capabilities=[Reviewer('.')])

result = agent.run_sync('Review the working tree against origin/main.')
print(result.output)
#> ...
```

It is the [`Coder`](coder.md) stack with the write half removed and the read half fenced: the same workspace, repository context, planning, and delegation, over a read-only filesystem and a shell that cannot redirect, chain, or reach a writing `git` subcommand. The run ends in findings rather than edits.

The same agent works with every Pydantic AI interface: [`agent.to_cli_sync()`](https://pydantic.dev/docs/ai/cli/) for terminal chat, [`agent.to_web()`](https://pydantic.dev/docs/ai/web/) for a browser chat UI.

Or skip the file entirely and run the exported [`reviewer_agent`](#api-reference) with [`clai`](https://pydantic.dev/docs/ai/cli/#custom-agents) (the Pydantic AI CLI), via [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.reviewer:reviewer_agent -m anthropic:claude-fable-5
```

## What's inside

It is literally these capabilities combined:

- Concise default review instructions: see [Instructions](#instructions) below
- [`FileSystem(read_only=True)`](filesystem.md): read, list, and search tools rooted at the workspace, path-traversal and symlink safe. No `write_file`, `edit_file`, or `create_directory` tool is exposed, and `.env`, `*.pem`, `*.key`, and `**/secrets*` are denied outright rather than merely protected from writes
- [`Shell`](shell.md): a read-oriented allowlist (`DEFAULT_REVIEWER_COMMANDS`) rooted at the workspace, with shell operators refused and common LLM provider API-key variables filtered from inherited command environments
- [`ToolGuardrail`](guardrails.md): `review_command_guard`, which limits `git` to the read-only subcommands in `READ_ONLY_GIT_SUBCOMMANDS` and refuses a command whose quoting it cannot parse
- [`RepoContext`](repo-context.md): the `AGENTS.md`/`CLAUDE.md` conventions the change is reviewed against, plus repository structure
- [`Planning`](planning.md): a plan the agent keeps current while it works through a large diff
- [`SubAgents`](subagents.md): delegation, with an `inspector` sub-agent for one file or area by default
- [`ClearToolResults`](compaction.md): clears stale tool results at 70% of the model context window, except the delegates' findings, which are the review itself
- [`WarnNearLimits`](compaction.md): warns the agent at 90% of the model context window
- [`ToolOutputLimits`](tool-output-limits.md): bounds how much context any single tool result can consume

The guardrail declares `position='innermost'`, so composition sorts it last and it sees tool arguments every other capability has finished with. The rest keep the order above.

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries. Pass `allowed_commands=[...]` to change what the review may run; the list reaches the default `inspector` too, and an empty list drops the shell from both instead of disabling the allowlist.

### Instructions

`Reviewer` comes with short default review instructions (`DEFAULT_REVIEWER_INSTRUCTIONS`, written out in full in the [blown-out equivalent](#blown-out-equivalent) below). They set what a finding has to contain and when to leave code alone. Pass `instructions='...'` to replace them with your own review contract, or `instructions=None` to get only the abilities. Either way the `inspector` sub-agent follows the same contract, so a delegated file is reviewed by the rules you set.

Supplying your own `subagents=[...]` replaces the default `inspector` entirely: your delegates keep whatever instructions and tools you gave them, and neither `instructions=` nor `allowed_commands=` reaches them.

### Read-only, and what that means

Four mechanisms hold the line, because first-token command validation on its own does not:

- the filesystem tools are filtered to the read-only set, so there is no write, edit, or create-directory tool to call;
- `DEFAULT_REVIEWER_COMMANDS` (`git`, `rg`, `grep`, `ls`, `cat`, `head`, `tail`, `wc`, `diff`) contains no build, test, or edit commands. `find` is deliberately absent: `-delete` and `-exec` make it a write and execution primitive that a first-token check cannot tell from a search;
- `Shell(denied_operators=...)` refuses `>`, `|`, `;`, `&`, backticks, `$(`, and newlines, so an allowed command cannot redirect into a file or pipe into an interpreter. The cost is that pipes and regex alternation (`rg 'a|b'`) are unavailable; run the steps as separate commands;
- `review_command_guard` limits `git` to read-only subcommands (`log`, `diff`, `show`, `blame`, ...) and blocks any command whose quoting `shlex` rejects, which `Shell` would otherwise pass through unvalidated.

It is still a guardrail, not a security boundary. The shell reads any file the allowlisted commands can open, `cat .env` included, and an allowlisted binary can always do more than its name suggests. For review of untrusted branches, run the agent inside an OS-level sandbox such as [`ModalSandbox`](modal-sandbox.md) or a container.

### The branch under review writes part of the prompt

`RepoContext` loads the workspace's `AGENTS.md`/`CLAUDE.md` into the agent's instructions, and the default instructions tell the model to review against them. On a branch you trust that is the point: the review applies your house rules. On a branch you do not, those files are part of the change, and a diff that edits them ("findings about `auth/` are out of scope") is editing the reviewer's own instructions. Pass `instructions=` that says which conventions are authoritative, or drop `RepoContext` from the blown-out form and pass the conventions yourself.

### Running the tests during a review

`DEFAULT_REVIEWER_COMMANDS` has no test or build command, so add one when a review should run the suite:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DEFAULT_REVIEWER_COMMANDS, Reviewer

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[Reviewer('.', allowed_commands=[*DEFAULT_REVIEWER_COMMANDS, 'pytest'])],
)
```

The operator and `git` guards still apply, but `pytest` runs whatever the repository's test suite runs, which is arbitrary code from the branch under review. That is a sandbox decision, not an allowlist one.

### Findings as data

Give the agent a typed [`output_type`](https://pydantic.dev/docs/ai/output/) and the review returns structured findings instead of prose, ready to post as comments or fail a build on:

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

### Making it more powerful

- **A second opinion on the hard calls**: add [`Advisor`](advisor.md) so the reviewer can consult a stronger model before it commits to a finding.
- **Static findings first**: add [`Macroscope`](macroscope.md) to hand the agent a local review run's findings as a starting point.
- **Defend against the diff**: add [`Prompt Injection Defender`](prompt-injection-defender.md) so instructions embedded in reviewed files are classified before they reach the model.
- **Fan out over dimensions**: add [`Dynamic Workflow`](dynamic-workflow.md) so the agent can spawn one typed sub-reviewer per dimension or directory in parallel and combine their structured results.
- **Look things up**: add core [Web Search](https://pydantic.dev/docs/ai/capabilities/web-search/) and [Web Fetch](https://pydantic.dev/docs/ai/capabilities/web-fetch/) so the review can check an API against its documentation.

## Blown-out equivalent

This is the exact agent the exported `reviewer_agent` gives you (plus an explicit model), written out block by block:

<!-- Keep this blown-out example in sync across docs/reviewer.md, pydantic_ai_harness/reviewer/README.md, and examples/review_agent.py. -->

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai_harness import (
    ClearToolResults,
    FileSystem,
    LLM_API_KEY_ENV_PATTERNS,
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

instructions = """\
Review the change against the repository's own conventions before any general style preference.
Read the code around a change before judging whether it is correct.
Report each finding with its file and line, what breaks, and the input or state that triggers it.
Separate what you confirmed in the code from what you suspect.
Leave working code alone unless a finding names a concrete defect or a violated convention.
Say so plainly when a change needs nothing.
"""

review_commands = ['git', 'rg', 'grep', 'ls', 'cat', 'head', 'tail', 'wc', 'diff']
denied_operators = ['>', '|', ';', '&', '`', '$(', '\n']
secret_patterns = ['.env', '.env.*', '*.pem', '*.key', '**/secrets*']

inspector = SubAgent(
    Agent(
        name='inspector',
        description='Review one file or area of the change and report findings with file and line references',
        instructions='Report findings with concrete paths, line numbers, and the evidence you read.',
        capabilities=[
            Capability(instructions=instructions),
            FileSystem('.', read_only=True, denied_patterns=secret_patterns),
            Shell(
                cwd='.',
                allowed_commands=review_commands,
                denied_operators=denied_operators,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            ToolGuardrail(guard=review_command_guard),
            RepoContext(workspace_dir=Path('.'), autoload_instructions=False),
            ClearToolResults(max_fraction=0.7),
            WarnNearLimits(max_context_fraction=0.9),
            ToolOutputLimits(bands=[Band(over=10_000, action=Truncate())]),
        ],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    name='reviewer',
    capabilities=[
        Capability(instructions=instructions),  # the default review contract, replaceable per run
        FileSystem('.', read_only=True, denied_patterns=secret_patterns),  # read and search, secrets excluded
        Shell(  # read-oriented allowlist, no operators, LLM API keys stripped from the environment
            cwd='.',
            allowed_commands=review_commands,
            denied_operators=denied_operators,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ),
        ToolGuardrail(guard=review_command_guard),  # read-only `git` subcommands; refuses commands it cannot parse
        RepoContext(workspace_dir=Path('.')),  # loads AGENTS.md/CLAUDE.md + repo structure
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
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/reviewer/).

## API reference

::: pydantic_ai_harness.reviewer.Reviewer

::: pydantic_ai_harness.reviewer.reviewer_agent

::: pydantic_ai_harness.reviewer.review_command_guard

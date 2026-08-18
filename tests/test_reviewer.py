import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import READ_ONLY_TOOL_NAMES, FileSystem
from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolGuardrail
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.reviewer import (
    DEFAULT_REVIEWER_COMMANDS,
    DEFAULT_REVIEWER_INSTRUCTIONS,
    DENIED_SHELL_OPERATORS,
    SECRET_PATH_PATTERNS,
    Reviewer,
    review_command_guard,
    reviewer_agent,
)
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import Spill, ToolOutputLimits, Truncate

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _shell(reviewer: Reviewer[None]) -> Shell[None]:
    return next(capability for capability in reviewer.capabilities if isinstance(capability, Shell))


def _delegate_capabilities(reviewer: Reviewer[None]) -> list[AbstractCapability[None]]:
    subagents = next(capability for capability in reviewer.capabilities if isinstance(capability, SubAgents))
    return list(subagents.agents[0].agent.root_capability.capabilities)


def _tool_returns(messages: list[ModelMessage]) -> str:
    """Every tool result in the run, joined, as the model saw them.

    A refused tool surfaces as a retry prompt rather than a return, so both count.
    """
    return ' '.join(
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    )


def _calls_then_reports(tool_name: str, **args: str) -> FunctionModel:
    """A model that makes one tool call, then reports whatever came back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id='call-1')])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(respond)


def _guard(command: str, *, tool: str = 'run_command') -> GuardrailResult:
    return review_command_guard(ToolCallInfo(name=tool, args={'command': command}, tool_call_id='call-1'))


def test_reviewer_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Reviewer()])

    assert isinstance(agent, Agent)


def test_reviewer_agent_is_model_less_and_composed() -> None:
    assert isinstance(reviewer_agent, Agent)
    assert reviewer_agent.model is None
    assert reviewer_agent.name == 'reviewer'


def test_reviewer_agent_export_is_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import pydantic_ai_harness.reviewer; '
            "assert 'pydantic_ai_harness.reviewer._agent' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_reviewer_unknown_export() -> None:
    import pydantic_ai_harness.reviewer

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.reviewer.__getattr__('missing')


def test_reviewer_members_are_transparent(tmp_path: Path) -> None:
    reviewer = Reviewer[None](tmp_path)

    # `ToolGuardrail` declares `position='innermost'`, so composition sorts it last.
    assert [type(capability) for capability in reviewer.capabilities] == [
        Capability,
        FileSystem,
        Shell,
        RepoContext,
        Planning,
        SubAgents,
        ClearToolResults,
        WarnNearLimits,
        ToolOutputLimits,
        ToolGuardrail,
    ]
    capability = next(capability for capability in reviewer.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == [DEFAULT_REVIEWER_INSTRUCTIONS]


def test_reviewer_capabilities_are_rooted_at_the_workspace(tmp_path: Path) -> None:
    reviewer = Reviewer[None](tmp_path)

    file_system = next(capability for capability in reviewer.capabilities if isinstance(capability, FileSystem))
    repo_context = next(capability for capability in reviewer.capabilities if isinstance(capability, RepoContext))
    assert file_system.root_dir == tmp_path
    assert repo_context.workspace_dir == tmp_path
    assert _shell(reviewer).cwd == tmp_path


def test_reviewer_expands_a_home_relative_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    reviewer = Reviewer[None]('~/checkout')

    file_system = next(capability for capability in reviewer.capabilities if isinstance(capability, FileSystem))
    repo_context = next(capability for capability in reviewer.capabilities if isinstance(capability, RepoContext))
    assert file_system.root_dir == tmp_path / 'checkout'
    assert repo_context.workspace_dir == tmp_path / 'checkout'
    assert _shell(reviewer).cwd == tmp_path / 'checkout'


def test_reviewer_withholds_secret_paths_from_the_file_tools(tmp_path: Path) -> None:
    reviewer = Reviewer[None](tmp_path)

    file_system = next(capability for capability in reviewer.capabilities if isinstance(capability, FileSystem))
    assert file_system.read_only is True
    # `protected_patterns` gates writes only, so under `read_only=True` the secrets it
    # names stay readable unless they are denied outright.
    assert tuple(file_system.denied_patterns) == SECRET_PATH_PATTERNS


async def test_reviewer_file_tools_refuse_to_read_secrets(tmp_path: Path) -> None:
    (tmp_path / '.env').write_text('OPENAI_API_KEY=sk-secret\n', encoding='utf-8')
    agent = Agent(_calls_then_reports('read_file', path='.env'), capabilities=[Reviewer(tmp_path)])

    result = await agent.run('summarize the configuration')

    rendered = _tool_returns(result.all_messages())
    assert 'sk-secret' not in rendered
    assert 'denied by pattern' in rendered


async def test_reviewer_file_tools_still_read_source(tmp_path: Path) -> None:
    (tmp_path / 'app.py').write_text('value = 1\n', encoding='utf-8')
    agent = Agent(_calls_then_reports('read_file', path='app.py'), capabilities=[Reviewer(tmp_path)])

    result = await agent.run('read the module')

    assert 'value = 1' in _tool_returns(result.all_messages())


def test_reviewer_default_commands_are_pinned() -> None:
    # Pinned, not spot-checked: changing this tuple must also revisit its docstring and
    # the four written-out copies of the composition.
    assert DEFAULT_REVIEWER_COMMANDS == ('git', 'rg', 'grep', 'ls', 'cat', 'head', 'tail', 'wc', 'diff')
    # `find -delete` and `find -exec` are writes that first-token validation cannot see.
    assert 'find' not in DEFAULT_REVIEWER_COMMANDS


def test_reviewer_shell_denies_operators_and_strips_provider_keys(tmp_path: Path) -> None:
    shell = _shell(Reviewer[None](tmp_path))

    assert tuple(shell.allowed_commands) == DEFAULT_REVIEWER_COMMANDS
    assert tuple(shell.denied_operators) == DENIED_SHELL_OPERATORS
    assert tuple(shell.denied_env_patterns) == LLM_API_KEY_ENV_PATTERNS


@pytest.mark.parametrize(
    'command',
    [
        'cat notes.txt > stolen.txt',
        'cat notes.txt >> stolen.txt',
        'ls && python3 -c "print(1)"',
        'ls; touch created.txt',
        'cat notes.txt | sh',
        'echo `touch created.txt`',
        'echo $(touch created.txt)',
        'ls\ntouch created.txt',
        'python3 -c "print(1)"',
        'sed -i s/a/b/ notes.txt',
        'find . -delete',
        'touch created.txt',
    ],
)
async def test_reviewer_shell_refuses_writes(tmp_path: Path, command: str) -> None:
    (tmp_path / 'notes.txt').write_text('original\n', encoding='utf-8')
    toolset = _shell(Reviewer[None](tmp_path)).get_toolset()

    with pytest.raises(ModelRetry):
        await toolset.run_command(command)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['notes.txt']
    assert (tmp_path / 'notes.txt').read_text(encoding='utf-8') == 'original\n'


async def test_reviewer_shell_still_reads(tmp_path: Path) -> None:
    (tmp_path / 'notes.txt').write_text('original\n', encoding='utf-8')
    toolset = _shell(Reviewer[None](tmp_path)).get_toolset()

    assert 'original' in await toolset.run_command('cat notes.txt')
    assert 'notes.txt' in await toolset.run_command('ls')


@pytest.mark.parametrize(
    'command',
    ['git commit -am wip', 'git push --force', 'git checkout -- .', 'git clean -fdx', 'git', 'git -C /tmp log'],
)
def test_review_command_guard_blocks_writing_git_commands(command: str) -> None:
    assert _guard(command).action == 'block'


@pytest.mark.parametrize('command', ['git log --oneline', 'git diff origin/main', 'git show HEAD', 'rg pattern'])
def test_review_command_guard_allows_reading_commands(command: str) -> None:
    assert _guard(command).action == 'allow'


def test_review_command_guard_blocks_commands_it_cannot_parse() -> None:
    # `shlex` rejects this quoting and `Shell` reads that as "nothing to validate",
    # so without the guard an unlisted command runs.
    assert _guard("touch created.txt #'").action == 'block'


def test_review_command_guard_ignores_other_tools() -> None:
    assert _guard('git commit -am wip', tool='read_file').action == 'allow'
    assert review_command_guard(ToolCallInfo(name='run_command', args={}, tool_call_id='c')).action == 'allow'


def test_reviewer_guards_the_shell_it_builds(tmp_path: Path) -> None:
    guardrail = next(
        capability for capability in Reviewer[None](tmp_path).capabilities if isinstance(capability, ToolGuardrail)
    )

    assert guardrail.guard is review_command_guard
    assert any(isinstance(capability, ToolGuardrail) for capability in _delegate_capabilities(Reviewer[None](tmp_path)))


def test_reviewer_empty_allowlist_removes_the_shell(tmp_path: Path) -> None:
    # `Shell` reads an empty allowlist as "no allowlist", so forwarding one would widen
    # the review agent instead of narrowing it.
    reviewer = Reviewer[None](tmp_path, allowed_commands=[])

    assert not any(isinstance(capability, (Shell, ToolGuardrail)) for capability in reviewer.capabilities)
    assert not any(isinstance(capability, (Shell, ToolGuardrail)) for capability in _delegate_capabilities(reviewer))
    assert any(isinstance(capability, FileSystem) for capability in reviewer.capabilities)


def test_reviewer_copies_the_caller_allowlist(tmp_path: Path) -> None:
    commands = ['git', 'rg']
    reviewer = Reviewer[None](tmp_path, allowed_commands=commands)
    commands.append('curl')

    assert tuple(_shell(reviewer).allowed_commands) == ('git', 'rg')
    delegate_shell = next(c for c in _delegate_capabilities(reviewer) if isinstance(c, Shell))
    assert tuple(delegate_shell.allowed_commands) == ('git', 'rg')


def test_reviewer_delegate_shares_the_review_tools(tmp_path: Path) -> None:
    subagents = next(c for c in Reviewer[None](tmp_path).capabilities if isinstance(c, SubAgents))
    delegate = subagents.agents[0].agent

    assert delegate.name == 'inspector'
    assert delegate.description == (
        'Review one file or area of the change and report findings with file and line references'
    )
    capabilities = list(delegate.root_capability.capabilities)
    file_system = next(c for c in capabilities if isinstance(c, FileSystem))
    assert file_system.read_only is True
    assert file_system.root_dir == tmp_path
    assert tuple(file_system.denied_patterns) == SECRET_PATH_PATTERNS
    shell = next(c for c in capabilities if isinstance(c, Shell))
    assert shell.cwd == tmp_path
    assert tuple(shell.denied_operators) == DENIED_SHELL_OPERATORS
    repo_context = next(c for c in capabilities if isinstance(c, RepoContext))
    assert repo_context.workspace_dir == tmp_path
    # The parent already carries the workspace instruction files.
    assert repo_context.autoload_instructions is False
    assert any(isinstance(c, ClearToolResults) for c in capabilities)
    assert any(isinstance(c, WarnNearLimits) for c in capabilities)


def test_reviewer_delegate_truncates_instead_of_spilling(tmp_path: Path) -> None:
    # A delegation's spill handles are keyed by its own run, so a spilled file is
    # unreadable once it reports back, and the store keeps it anyway.
    limits = next(c for c in _delegate_capabilities(Reviewer[None](tmp_path)) if isinstance(c, ToolOutputLimits))

    assert [type(band.action) for band in limits.bands] == [Truncate]


def test_reviewer_keeps_delegated_findings_out_of_clearing(tmp_path: Path) -> None:
    clearing = next(c for c in Reviewer[None](tmp_path).capabilities if isinstance(c, ClearToolResults))
    parent_limits = next(c for c in Reviewer[None](tmp_path).capabilities if isinstance(c, ToolOutputLimits))

    assert clearing.exclude_tools == frozenset({'delegate_task'})
    assert [type(band.action) for band in parent_limits.bands] == [Spill]


def test_reviewer_threads_instructions_to_the_delegate(tmp_path: Path) -> None:
    reviewer = Reviewer[None](tmp_path, instructions='Only report security defects.')

    capability = next(c for c in reviewer.capabilities if isinstance(c, Capability))
    assert capability.get_instructions() == ['Only report security defects.']
    delegate_instructions = next(c for c in _delegate_capabilities(reviewer) if isinstance(c, Capability))
    assert delegate_instructions.get_instructions() == ['Only report security defects.']


def test_reviewer_none_disables_instructions(tmp_path: Path) -> None:
    reviewer = Reviewer[None](tmp_path, instructions=None)

    assert not any(isinstance(capability, Capability) for capability in reviewer.capabilities)
    assert not any(isinstance(capability, Capability) for capability in _delegate_capabilities(reviewer))


def test_reviewer_empty_subagents_disables_delegation() -> None:
    reviewer = Reviewer[None](subagents=[])

    assert not any(isinstance(capability, SubAgents) for capability in reviewer.capabilities)


def test_reviewer_custom_subagents_replace_the_default() -> None:
    delegate = SubAgent(Agent(name='security', description='Review the change for security defects'))
    reviewer = Reviewer[None](subagents=[delegate])

    subagents = next(capability for capability in reviewer.capabilities if isinstance(capability, SubAgents))
    assert [sub.agent.name for sub in subagents.agents] == ['security']


def test_reviewer_for_agent_preserves_subclass() -> None:
    reviewer = Reviewer[None]()
    bound = reviewer.for_agent(Agent(TestModel()))

    assert isinstance(bound, Reviewer)


async def test_reviewer_exposes_no_workspace_write_tools(tmp_path: Path) -> None:
    review_model = TestModel(call_tools=[])
    review_agent = Agent(review_model, capabilities=[Reviewer(tmp_path)])
    writable_model = TestModel(call_tools=[])
    writable_agent = Agent(writable_model, capabilities=[FileSystem(tmp_path)])

    await review_agent.run('review the change')
    await writable_agent.run('edit the change')

    assert review_model.last_model_request_parameters is not None
    assert writable_model.last_model_request_parameters is not None
    exposed = {tool.name for tool in review_model.last_model_request_parameters.function_tools}
    # The probe set is whatever `FileSystem` exposes when it is writable, not a hand-written
    # list, so a write tool added later cannot slip through unprobed.
    file_tools = {tool.name for tool in writable_model.last_model_request_parameters.function_tools}
    assert {'write_file', 'edit_file', 'create_directory'} <= file_tools
    assert exposed & file_tools <= READ_ONLY_TOOL_NAMES
    assert 'read_file' in exposed
    assert 'run_command' in exposed

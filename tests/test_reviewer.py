from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import READ_ONLY_TOOL_NAMES, FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.reviewer import (
    DEFAULT_REVIEW_COMMANDS,
    DEFAULT_REVIEWER_INSTRUCTIONS,
    Reviewer,
    reviewer_agent,
)
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_reviewer_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Reviewer()])

    assert isinstance(agent, Agent)


def test_reviewer_agent_is_model_less_and_composed() -> None:
    assert isinstance(reviewer_agent, Agent)
    assert reviewer_agent.model is None
    assert reviewer_agent.name == 'reviewer'
    assert any(isinstance(capability, FileSystem) for capability in reviewer_agent.root_capability.capabilities)


def test_reviewer_unknown_export() -> None:
    import pydantic_ai_harness.reviewer

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.reviewer.__getattr__('missing')


def test_reviewer_members_are_transparent(tmp_path: Path) -> None:
    reviewer = Reviewer(tmp_path)

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
    ]
    capability = next(capability for capability in reviewer.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == [DEFAULT_REVIEWER_INSTRUCTIONS]
    file_system = next(capability for capability in reviewer.capabilities if isinstance(capability, FileSystem))
    assert file_system.read_only is True
    assert file_system.root_dir == tmp_path
    shell = next(capability for capability in reviewer.capabilities if isinstance(capability, Shell))
    assert tuple(shell.allowed_commands) == DEFAULT_REVIEW_COMMANDS
    assert tuple(shell.denied_env_patterns) == LLM_API_KEY_ENV_PATTERNS
    repo_context = next(capability for capability in reviewer.capabilities if isinstance(capability, RepoContext))
    assert repo_context.workspace_dir == tmp_path


def test_reviewer_default_commands_are_read_oriented() -> None:
    # A write or build command in the default allowlist would contradict what the harness promises.
    assert not {'sed', 'python', 'uv', 'make', 'pytest', 'ruff'} & set(DEFAULT_REVIEW_COMMANDS)


def test_reviewer_delegate_shares_the_review_tools(tmp_path: Path) -> None:
    subagents = next(capability for capability in Reviewer(tmp_path).capabilities if isinstance(capability, SubAgents))
    delegate = subagents.agents[0].agent

    assert delegate.name == 'inspector'
    assert (
        delegate.description
        == 'Review one file or area of the change and report findings with file and line references'
    )
    delegate_capabilities = delegate.root_capability.capabilities
    delegate_file_system = next(c for c in delegate_capabilities if isinstance(c, FileSystem))
    assert delegate_file_system.read_only is True
    assert any(isinstance(capability, RepoContext) for capability in delegate_capabilities)
    assert any(isinstance(capability, ToolOutputLimits) for capability in delegate_capabilities)


def test_reviewer_threads_allowed_commands_to_delegate(tmp_path: Path) -> None:
    reviewer = Reviewer(tmp_path, allowed_commands=['git', 'pytest'])

    shell = next(capability for capability in reviewer.capabilities if isinstance(capability, Shell))
    assert list(shell.allowed_commands) == ['git', 'pytest']
    subagents = next(capability for capability in reviewer.capabilities if isinstance(capability, SubAgents))
    delegate_shell = next(c for c in subagents.agents[0].agent.root_capability.capabilities if isinstance(c, Shell))
    assert list(delegate_shell.allowed_commands) == ['git', 'pytest']


def test_reviewer_threads_instructions() -> None:
    reviewer = Reviewer(instructions='Custom instructions')

    capability = next(capability for capability in reviewer.capabilities if isinstance(capability, Capability))
    assert capability.get_instructions() == ['Custom instructions']


def test_reviewer_none_disables_instructions() -> None:
    reviewer = Reviewer(instructions=None)

    assert not any(isinstance(capability, Capability) for capability in reviewer.capabilities)


def test_reviewer_empty_subagents_disables_delegation() -> None:
    reviewer = Reviewer(subagents=[])

    assert not any(isinstance(capability, SubAgents) for capability in reviewer.capabilities)


def test_reviewer_custom_subagents_replace_the_default() -> None:
    delegate = SubAgent(Agent(name='security', description='Review the change for security defects'))
    reviewer = Reviewer(subagents=[delegate])

    subagents = next(capability for capability in reviewer.capabilities if isinstance(capability, SubAgents))
    assert [sub.agent.name for sub in subagents.agents] == ['security']


def test_reviewer_for_agent_preserves_subclass() -> None:
    reviewer = Reviewer()
    bound = reviewer.for_agent(Agent(TestModel()))

    assert isinstance(bound, Reviewer)


async def test_reviewer_exposes_no_workspace_write_tools(tmp_path: Path) -> None:
    model = TestModel(call_tools=[])
    agent = Agent(model, capabilities=[Reviewer(tmp_path)])

    await agent.run('review the change')

    assert model.last_model_request_parameters is not None
    tool_names = {tool.name for tool in model.last_model_request_parameters.function_tools}
    file_tools = tool_names & {'read_file', 'write_file', 'edit_file', 'delete_file', 'search_files'}
    assert file_tools <= READ_ONLY_TOOL_NAMES
    assert 'read_file' in tool_names
    assert 'run_command' in tool_names

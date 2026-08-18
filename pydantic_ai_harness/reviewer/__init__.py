"""Complete code-review harness."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.reviewer._capability import (
    DEFAULT_REVIEWER_COMMANDS,
    DEFAULT_REVIEWER_INSTRUCTIONS,
    DENIED_SHELL_OPERATORS,
    READ_ONLY_GIT_SUBCOMMANDS,
    SECRET_PATH_PATTERNS,
    SHELL_COMMAND_TOOLS,
    Reviewer,
    review_command_guard,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.reviewer._agent import reviewer_agent

__all__ = [
    'DEFAULT_REVIEWER_COMMANDS',
    'DEFAULT_REVIEWER_INSTRUCTIONS',
    'DENIED_SHELL_OPERATORS',
    'READ_ONLY_GIT_SUBCOMMANDS',
    'SECRET_PATH_PATTERNS',
    'SHELL_COMMAND_TOOLS',
    'Reviewer',
    'review_command_guard',
    'reviewer_agent',
]


def __getattr__(name: str) -> object:
    if name == 'reviewer_agent':
        from pydantic_ai_harness.reviewer._agent import reviewer_agent

        return reviewer_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

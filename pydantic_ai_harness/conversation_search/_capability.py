"""The `ConversationSearch` capability: BM25 recall over persisted step history."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.conversation_search._source import HistorySource
from pydantic_ai_harness.conversation_search._toolset import ConversationSearchToolset, SearchScope

_INSTRUCTIONS_PREFIX = (
    'A `search_conversation_history` tool can retrieve exact details from persisted history: '
    'earlier turns that context compaction has since dropped from the live context, and past '
)
_INSTRUCTIONS_SUFFIX = ' Reach for it when the current context, or a compaction summary, lacks a detail you need.'

_CONVERSATION_INSTRUCTIONS = f'{_INSTRUCTIONS_PREFIX}runs in the same conversation.{_INSTRUCTIONS_SUFFIX}'
"""Instruction text under `scope='conversation'`, where the corpus stops at the conversation."""

_ALL_INSTRUCTIONS = f'{_INSTRUCTIONS_PREFIX}runs persisted in the same store.{_INSTRUCTIONS_SUFFIX}'
"""Instruction text under `scope='all'`. Naming the conversation boundary here would talk the
model out of the cross-conversation recall that opting into `all` exists to enable."""


@dataclass
class ConversationSearch(AbstractCapability[AgentDepsT]):
    """Search persisted conversation history with a dependency-free BM25 tool.

    This capability persists nothing itself: it reads whatever history a persistence
    capability already stores, through a `HistorySource`. Pair it with
    `StepPersistence` sharing the same store, and the model can recall what
    compaction dropped from the live context as well as anything from past runs in
    the same conversation:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.compaction import SlidingWindowCompaction
    from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource
    from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

    store = SqliteStepStore(database='sessions.db')
    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            StepPersistence(store=store),
            ConversationSearch(SnapshotHistorySource(store)),
            SlidingWindowCompaction(max_messages=40),
        ],
    )


    async def ask(question: str, conversation_id: str) -> str:
        result = await agent.run(question, conversation_id=conversation_id)
        return result.output
    ```

    Search is conversation-scoped by default, so reaching a past run requires both
    runs to share a `conversation_id`. pydantic-ai resolves one per run: an explicit
    `conversation_id=` wins, otherwise the most recent `conversation_id` on
    `message_history` is inherited, otherwise a fresh one is generated. Threading
    `message_history` through follow-up runs keeps them in one conversation; runs
    sharing neither an explicit id nor a history chain are separate.

    Some compaction strategies persist their edits into the run's durable message
    history (`SummarizingCompaction` replaces summarized prefixes for good; a
    `SlidingWindowCompaction` trim only narrows what each request sends). Either way,
    `StepPersistence` snapshots each step boundary before the next compaction runs,
    so the union of a run's snapshots still holds the originals --
    `SnapshotHistorySource` recovers them. No ordering or hook coordination between
    the capabilities is required; the search tool reads the store lazily at call
    time.
    """

    source: HistorySource
    """Where the search corpus comes from. Use `SnapshotHistorySource` over the
    store a `StepPersistence` capability writes to."""

    scope: SearchScope = 'conversation'
    """How much of the store one search may reach.

    `conversation` restricts the corpus to runs whose `conversation_id` matches the
    calling run's. A run with no `conversation_id` searches nothing and the tool says
    so, rather than falling back to every other unlabelled run. `all` searches every
    run the source enumerates and must only be used when the store is already isolated
    to one principal.
    """

    tool_id: str = 'conversation-search'
    """Toolset id for the `search_conversation_history` tool."""

    max_matches: int = 10
    """Maximum number of matching excerpts the search tool returns. Must be non-negative."""

    context_lines: int = 5
    """Number of surrounding lines shown around each search match. Must be non-negative."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation, non-negative. This capability's default; Lucene's
    `BM25Similarity` uses `1.2`."""

    bm25_b: float = 0.75
    """BM25 length-normalization, between `0.0` and `1.0` (Lucene/Elasticsearch default)."""

    add_instructions: bool = True
    """Emit a short instruction telling the model the recall tool exists."""

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_conversation_history` tool over the source."""
        return ConversationSearchToolset[AgentDepsT](
            self.source,
            tool_id=self.tool_id,
            max_matches=self.max_matches,
            context_lines=self.context_lines,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            scope=self.scope,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model the recall tool exists, unless `add_instructions` is false.

        The wording follows `scope` so it never describes a corpus narrower than the
        one configured.
        """
        if not self.add_instructions:
            return None
        if self.scope == 'conversation':
            return _CONVERSATION_INSTRUCTIONS
        return _ALL_INSTRUCTIONS

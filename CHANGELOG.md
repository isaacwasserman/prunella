# prunella

## 0.2.0

### Minor Changes

- 31d66ff: Add compactor lifecycle hooks (onCompactStart, onCompactEnd, onSummaryCreate, onSummaryMerge) for observing and reacting to compaction events.
- 7ee2948: Initial release of Prunella: policy-driven pruning of AI SDK conversation messages, with a `recall-pruned` tool that restores a pruned part for a bounded duration.
- 2fd5577: Replace top-level firstPartId/lastPartId/collapsed on CompactorSummary with a spans array so a single summary can cover multiple non-consecutive regions. Eliminates the collapsed sentinel hack in favor of honest multi-span representation.
- c758c43: Restructure pruning and compaction into a parallel pipeline with stable part IDs. Pruning now returns a mask instead of mutated messages, and the recall tool returns original content via its tool output rather than rewriting history. A new render step composes both transformations. Adds compaction with LLM-driven summarization, configurable policies, and bounded iteration.
- fb7aa51: Compaction now works with pruning instead of beside it, and keeps one rolling summary.

  - Rendered output is always a valid prompt. A pruned tool call or result keeps its type, so every result still has its call, and a message with nothing pruned or summarized is passed through unchanged (system messages keep their string content).
  - Compaction measures the conversation as it will be sent, with pruned parts as placeholders and summaries in place, and a span counts toward `minCompactableSpan` by what it saves in that view. Summaries are still written from the original parts.
  - New `keepRecent` compaction option (a `PartAge`, like `olderThan`). Parts this recent are never compacted. The default, `{ turns: 0 }`, keeps the latest user message and everything after it.
  - One rolling summary: each compaction rewrites the summary from the previous one plus the newly eligible turns, and the new summary replaces the old one. Sessions that already have several summaries have them merged into one.
  - `onCompactStart` fires only when compaction is about to summarize or merge, and can return `false` to skip compaction for that call. `onCompactEnd` fires only after a start and reports `summariesCreated` and `summariesMerged`. `onSummaryCreate` reports `replacedSummaries`, and `onSummaryMerge` takes any number of `sourceSummaries`.
  - `prepare` returns `recall-pruned` only when a part was pruned. The compactor's summary recall tool is removed for now, because a recall of the rolling summary returns all compacted history.
  - The summarizer transcript labels a tool result's content `output` instead of `toolInput`.

- bc9782c: Add generic TRuntimeConfig support to Compactor, CompactorStore, and Prunella, allowing callers to pass per-request configuration through to store methods.
- ccaa232: Add optional summaryPrompt parameter to Compactor and Prunella that lets users inject custom instructions into summarization prompts.

### Patch Changes

- fb7aa51: Shorten pruning IDs from 64-char SHA-256 hex to 8-char nanoid-style IDs. The IDs stay deterministic.

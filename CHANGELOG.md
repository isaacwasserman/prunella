# prunella

## 0.3.0

### Minor Changes

- 1430e24: The `@tokenxl/count` peer dependency is now `^0.1.0`, its first stable release. Install `@tokenxl/count@^0.1.0` with this version.

## 0.2.0

### Minor Changes

- 0fdfbbd: `prepare` takes an optional `tools` set: the tools that you send with the messages. Their definitions add to the token count of the whole request, so `hasPressure` budgets, `compactionThreshold`, and the `estimatedTokens` of the compaction hooks include them. Prunella does not change or return these tools. When a part is pruned, these counts also include the `recall-pruned` tool that `prepare` returns, and `hasPressure` prunes enough to make space for that tool.
- 7c965c6: Compaction keeps a safer recent tail.

  - A tokens `keepRecent` counts the parts after a part as rendered, so a pruned part takes only its placeholder's size of the tail.
  - Compaction summarizes a tool call and its result together or keeps both, so a tail that ends inside a step never leaves a result without its call.
  - Compaction never summarizes the latest user message, even when `keepRecent` is shorter than a turn. The summary of earlier steps of the current turn is rendered after that message.

- 31d66ff: Add compactor lifecycle hooks (onCompactStart, onCompactEnd, onSummaryCreate, onSummaryMerge) for observing and reacting to compaction events.
- 7ee2948: Initial release of Prunella: policy-driven pruning of AI SDK conversation messages, with a `recall-pruned` tool that restores a pruned part for a bounded duration.
- 2fd5577: Replace top-level firstPartId/lastPartId/collapsed on CompactorSummary with a spans array so a single summary can cover multiple non-consecutive regions. Eliminates the collapsed sentinel hack in favor of honest multi-span representation.
- c758c43: Restructure pruning and compaction into a parallel pipeline with stable part IDs. Pruning now returns a mask instead of mutated messages, and the recall tool returns original content via its tool output rather than rewriting history. A new render step composes both transformations. Adds compaction with LLM-driven summarization, configurable policies, and bounded iteration.
- 7c965c6: Prune under pressure with a new `hasPressure` policy condition, and count part age in messages.

  - `{ hasPressure: { budget, bufferFactor? } }` is true for the parts chosen to relieve pressure. When the rendered conversation (summaries in place) is over `budget` tokens, the oldest parts that the rest of the policy would prune with the condition true are chosen, in steps of `bufferFactor × budget` tokens (`bufferFactor` defaults to 0.1). The conversation drops to about `budget × (1 − bufferFactor)` and grows back to `budget` before more is chosen, and the choice depends only on the history, so it is the same on every call. Several `hasPressure` conditions are resolved in the order they appear. Under `NOT` the condition is well defined, but the budget is no longer guaranteed.
  - `olderThan: { messages: n }` counts the messages after a part's message, so `{ messages: 1 }` keeps the latest assistant message and its tool results. It also works for compaction's `keepRecent`.

- 7c965c6: Export the public types from the package: `PruningPolicy`, `PartAge`, `PressureCondition`, `CompactionOptions`, `CompactorStore`, `CompactorSummary`, `PartSpan`, `CompactorHooks`, and `RuntimeConfig`.
- fb7aa51: Compaction now works with pruning instead of beside it, and keeps one rolling summary.

  - Rendered output is always a valid prompt. A pruned tool call or result keeps its type, so every result still has its call, and a message with nothing pruned or summarized is passed through unchanged (system messages keep their string content).
  - Compaction measures the conversation as it will be sent, with pruned parts as placeholders and summaries in place, and a span counts toward `minCompactableSpan` by what it saves in that view. Summaries are still written from the original parts.
  - New `keepRecent` compaction option (a `PartAge`, like `olderThan`). Parts this recent are never compacted. The default, `{ turns: 0 }`, keeps the latest user message and everything after it.
  - One rolling summary: each compaction rewrites the summary from the previous one plus the newly eligible turns, and the new summary replaces the old one. Sessions that already have several summaries have them merged into one.
  - `onCompactStart` fires only when compaction is about to summarize or merge, and can return `false` to skip compaction for that call. `onCompactEnd` fires only after a start and reports `summariesCreated` and `summariesMerged`. `onSummaryCreate` reports `replacedSummaries`, and `onSummaryMerge` takes any number of `sourceSummaries`.
  - `prepare` returns `recall-pruned` only when a part was pruned. The compactor's summary recall tool is removed for now, because a recall of the rolling summary returns all compacted history.
  - The summarizer transcript labels a tool result's content `output` instead of `toolInput`.

- bc9782c: Add generic TRuntimeConfig support to Compactor, CompactorStore, and Prunella, allowing callers to pass per-request configuration through to store methods.
- 7c965c6: Part IDs hash only content: the text of a text or reasoning part, and the call ID and tool name of a tool call or result. Provider fields and the form of a tool's input and output no longer change an ID, so a conversation keeps its IDs when it is stored and converted again. IDs from earlier versions do not match, so the spans of summaries stored by an earlier version are dropped and their text is carried into the next summary.
- ccaa232: Add optional summaryPrompt parameter to Compactor and Prunella that lets users inject custom instructions into summarization prompts.
- 7c965c6: `olderThan: { tokens: n }` is true when more than `n` tokens of parts come after a part, counted by their original content. It also works for compaction's `keepRecent`.
- 42e1dad: `Prunella` needs an `estimator` from `@tokenxl/count` (for example `createUsageEstimator("anthropic/claude-sonnet-5.5")`). Token counts use the estimator's whole-request count of the message array, not a text count of its JSON. Pressure budgets, `compactionThreshold`, and tokens part ages now include message overhead and use the model's profile. `@tokenxl/count` is a peer dependency, and `tokenx` is removed.

### Patch Changes

- 7c965c6: When compaction writes or merges a summary, `hasPressure` parts are chosen again on the summarized conversation. Parts that the summary now covers no longer cause more pruning, and `recall-pruned` is offered only when a part is still pruned.
- fb7aa51: Shorten pruning IDs from 64-char SHA-256 hex to 8-char nanoid-style IDs. The IDs stay deterministic.
- 7c965c6: A summary span whose first or last part is not in the conversation is dropped instead of failing `prepare`. The parts it covered are sent as they are, and the summary's text is carried into the next summary.

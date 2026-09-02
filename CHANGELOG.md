# prunella

## 0.2.0

### Minor Changes

- 31d66ff: Add compactor lifecycle hooks (onCompactStart, onCompactEnd, onSummaryCreate, onSummaryMerge) for observing and reacting to compaction events.
- 7ee2948: Initial release of Prunella: policy-driven pruning of AI SDK conversation messages, with a `recall-pruned` tool that restores a pruned part for a bounded duration.
- 2fd5577: Replace top-level firstPartId/lastPartId/collapsed on CompactorSummary with a spans array so a single summary can cover multiple non-consecutive regions. Eliminates the collapsed sentinel hack in favor of honest multi-span representation.
- c758c43: Restructure pruning and compaction into a parallel pipeline with stable part IDs. Pruning now returns a mask instead of mutated messages, and the recall tool returns original content via its tool output rather than rewriting history. A new render step composes both transformations. Adds compaction with LLM-driven summarization, configurable policies, and bounded iteration.
- bc9782c: Add generic TRuntimeConfig support to Compactor, CompactorStore, and Prunella, allowing callers to pass per-request configuration through to store methods.
- ccaa232: Add optional summaryPrompt parameter to Compactor and Prunella that lets users inject custom instructions into summarization prompts.

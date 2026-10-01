---
"prunella": minor
---

Compaction now works with pruning instead of beside it, and keeps one rolling summary.

- Rendered output is always a valid prompt. A pruned tool call or result keeps its type, so every result still has its call, and a message with nothing pruned or summarized is passed through unchanged (system messages keep their string content).
- Compaction measures the conversation as it will be sent, with pruned parts as placeholders and summaries in place, and a span counts toward `minCompactableSpan` by what it saves in that view. Summaries are still written from the original parts.
- New `keepRecent` compaction option (a `PartAge`, like `olderThan`). Parts this recent are never compacted. The default, `{ turns: 0 }`, keeps the latest user message and everything after it.
- One rolling summary: each compaction rewrites the summary from the previous one plus the newly eligible turns, and the new summary replaces the old one. Sessions that already have several summaries have them merged into one.
- `onCompactStart` fires only when compaction is about to summarize or merge, and can return `false` to skip compaction for that call. `onCompactEnd` fires only after a start and reports `summariesCreated` and `summariesMerged`. `onSummaryCreate` reports `replacedSummaries`, and `onSummaryMerge` takes any number of `sourceSummaries`.
- `prepare` returns `recall-pruned` only when a part was pruned. The compactor's summary recall tool is removed for now, because a recall of the rolling summary returns all compacted history.
- The summarizer transcript labels a tool result's content `output` instead of `toolInput`.

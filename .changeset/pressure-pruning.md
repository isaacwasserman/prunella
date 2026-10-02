---
"prunella": minor
---

Prune under pressure with a new `hasPressure` policy condition, and count part age in messages.

- `{ hasPressure: { budget, bufferFactor? } }` is true for the parts chosen to relieve pressure. When the rendered conversation (summaries in place) is over `budget` tokens, the oldest parts that the rest of the policy would prune with the condition true are chosen, in steps of `bufferFactor × budget` tokens (`bufferFactor` defaults to 0.1). The conversation drops to about `budget × (1 − bufferFactor)` and grows back to `budget` before more is chosen, and the choice depends only on the history, so it is the same on every call. Several `hasPressure` conditions are resolved in the order they appear. Under `NOT` the condition is well defined, but the budget is no longer guaranteed.
- `olderThan: { messages: n }` counts the messages after a part's message, so `{ messages: 1 }` keeps the latest assistant message and its tool results. It also works for compaction's `keepRecent`.

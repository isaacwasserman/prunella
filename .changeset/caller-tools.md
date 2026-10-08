---
"prunella": minor
---

`prepare` takes an optional `tools` set: the tools that you send with the messages. Their definitions add to the token count of the whole request, so `hasPressure` budgets, `compactionThreshold`, and the `estimatedTokens` of the compaction hooks include them. Prunella does not change or return these tools. When a part is pruned, these counts also include the `recall-pruned` tool that `prepare` returns, and `hasPressure` prunes enough to make space for that tool.

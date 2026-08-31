---
"prunella": minor
---

Restructure pruning and compaction into a parallel pipeline with stable part IDs. Pruning now returns a mask instead of mutated messages, and the recall tool returns original content via its tool output rather than rewriting history. A new render step composes both transformations. Adds compaction with LLM-driven summarization, configurable policies, and bounded iteration.

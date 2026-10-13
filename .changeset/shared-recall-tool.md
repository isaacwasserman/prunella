---
"prunella": minor
---

Each Prunella instance now has one `recall-pruned` tool, in `prunella.tools`. `prepare` does not return `recall-pruned` now. Send `prunella.tools` with every request. Token counts always include `recall-pruned`, also when no part is pruned. The tool keeps the content of the 10,000 parts that were pruned most recently.

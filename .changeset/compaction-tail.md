---
"prunella": minor
---

Compaction keeps a safer recent tail.

- A tokens `keepRecent` counts the parts after a part as rendered, so a pruned part takes only its placeholder's size of the tail.
- Compaction summarizes a tool call and its result together or keeps both, so a tail that ends inside a step never leaves a result without its call.
- Compaction never summarizes the latest user message, even when `keepRecent` is shorter than a turn. The summary of earlier steps of the current turn is rendered after that message.

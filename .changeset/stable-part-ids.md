---
"prunella": minor
---

Part IDs hash only content: the text of a text or reasoning part, and the call ID and tool name of a tool call or result. Provider fields and the form of a tool's input and output no longer change an ID, so a conversation keeps its IDs when it is stored and converted again. IDs from earlier versions do not match, so the spans of summaries stored by an earlier version are dropped and their text is carried into the next summary.

---
"prunella": patch
---

A summary span whose first or last part is not in the conversation is dropped instead of failing `prepare`. The parts it covered are sent as they are, and the summary's text is carried into the next summary.

---
"prunella": minor
---

`Prunella` needs an `estimator` from `@tokenxl/count` (for example `createUsageEstimator("anthropic/claude-sonnet-5.5")`). Token counts use the estimator's whole-request count of the message array, not a text count of its JSON. Pressure budgets, `compactionThreshold`, and tokens part ages now include message overhead and use the model's profile. `@tokenxl/count` is a peer dependency, and `tokenx` is removed.

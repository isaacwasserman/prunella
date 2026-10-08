# prunella

Policy-driven pruning of AI SDK conversation messages, with a recall tool to restore pruned parts, and optional rolling compaction into summaries.

## Install

```bash
bun add prunella @tokenxl/count
```

## Usage

```ts
import { createUsageEstimator } from "@tokenxl/count";
import { Prunella } from "prunella";

const prunella = new Prunella({
	estimator: createUsageEstimator("anthropic/claude-sonnet-5.5"),
	pruningPolicy: {
		AND: [{ hasType: "tool-result" }, { olderThan: { turns: 1 } }],
	},
});

const { messages, tools } = await prunella.prepare({
	messages: conversation,
	tools: myTools, // optional. Prunella only counts their tokens.
	sessionId: "session-1",
	config: undefined,
});

await generateText({ model, messages, tools: { ...myTools, ...tools } });
```

`estimator` counts tokens. Make it with `createUsageEstimator` from [`@tokenxl/count`](https://github.com/isaacwasserman/tokenx/tree/main/packages/count), with the profile of the model that gets the messages. Prunella counts the whole message array as one request, so message overhead is in the count. A part's size is its share of that count.

`prepare` also takes the `tools` that you send with the messages. Prunella adds their definitions to the count of the whole request (the `hasPressure` budget, `compactionThreshold`, and the `estimatedTokens` of the hooks). Prunella does not change or return these tools. When a part is pruned, the count also includes the `recall-pruned` tool that `prepare` returns.

`prepare` replaces each pruned part with a placeholder that holds a `pruneId`. When a part is pruned, `tools` contains `recall-pruned`, which returns the original content of a part. Part IDs come from the content of the conversation, so the same conversation always gets the same IDs.

## Pruning policy

A policy is a tree of conditions. A part is pruned when the policy is true for it.

| Condition | True when |
| --- | --- |
| `{ AND: [...] }`, `{ OR: [...] }`, `{ NOT: policy }` | The usual logic of the sub-policies. |
| `{ hasType: "tool-result" }` | The part has this type. |
| `{ hasRole: "assistant" }` | The part's message has this role. |
| `{ olderThan: age }` | The part is older than `age` (see below). |
| `{ shouldPrune: (args) => boolean }` | Your predicate returns `true`. |
| `{ hasPressure: { budget, bufferFactor? } }` | The part is chosen to keep the rendered conversation within `budget` tokens. |

A part age is one of:

- `{ turns: n }`: more than `n` user turns come after the part.
- `{ steps: n }`: more than `n` parts come after the part.
- `{ messages: n }`: more than `n` messages come after the part's message.
- `{ tokens: n }`: more than `n` tokens of parts come after the part.

`hasPressure` chooses the oldest parts that the rest of the policy would prune, in steps of `bufferFactor × budget` tokens (`bufferFactor` defaults to 0.1). The conversation drops to about `budget × (1 − bufferFactor)` and grows back to `budget` before more parts are chosen.

## Compaction

```ts
const prunella = new Prunella({
	estimator,
	pruningPolicy,
	compaction: {
		enabled: true,
		store, // a CompactorStore that saves summaries for each session
		model, // the model that writes summaries
		policy: {
			compactionThreshold: 80_000,
			minCompactableSpan: 2000,
			maxIterations: 3,
			keepRecent: { turns: 0 },
		},
	},
});
```

When the rendered conversation is over `compactionThreshold` tokens, the compactor summarizes the older parts into one rolling summary. Each new summary replaces the earlier one and carries its content forward.

- `keepRecent` is a part age. Parts this recent are never summarized. A tokens `keepRecent` counts pruned parts at the size of their placeholders.
- The latest user message is never summarized.
- A tool call and its result are summarized together or kept together.
- `policy.canCompact` can keep more parts out of summaries.
- `compaction.summaryPrompt` adds instructions to the summary prompt.
- `compaction.hooks` (`onCompactStart`, `onCompactEnd`, `onSummaryCreate`, `onSummaryMerge`) let you watch or skip compaction.

## Development

```bash
bun install
bun test
bun run typecheck
bun run lint
```

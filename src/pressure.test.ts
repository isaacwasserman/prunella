import { describe, expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { estimateTokenCount } from "tokenx";
import type { CompactorStore, CompactorSummary } from "./compaction";
import { Prunella } from "./index";
import type { PruningPolicy } from "./pruning";
import { attachIdsToMessages } from "./utils";

const PLACEHOLDER_PREFIX = "This part of the message has been pruned";

/** One user turn with `steps` tool calls, each returning a large result. */
function longTurn(steps: number): ModelMessage[] {
	const messages: ModelMessage[] = [
		{ role: "user", content: "Show me everything you can do" },
	];
	for (let step = 0; step < steps; step++) {
		const toolCallId = `call-${step}`;
		messages.push(
			{
				role: "assistant",
				content: [
					{
						type: "tool-call",
						toolCallId,
						toolName: "lookup",
						input: { step },
					},
				],
			},
			{
				role: "tool",
				content: [
					{
						type: "tool-result",
						toolCallId,
						toolName: "lookup",
						output: { type: "text", value: `step ${step} `.repeat(300) },
					},
				],
			},
		);
	}
	return messages;
}

function sizeOf(messages: ModelMessage[]): number {
	return estimateTokenCount(JSON.stringify(messages));
}

function prunedCallIds(messages: ModelMessage[]): string[] {
	return messages.flatMap((message) =>
		message.role === "tool"
			? message.content.flatMap((part) =>
					part.type === "tool-result" &&
					part.output.type === "text" &&
					part.output.value.startsWith(PLACEHOLDER_PREFIX)
						? [part.toolCallId]
						: [],
				)
			: [],
	);
}

function pressurePolicy(budget: number, bufferFactor?: number): PruningPolicy {
	return {
		AND: [
			{ hasType: "tool-result" },
			{ olderThan: { messages: 1 } },
			{
				hasPressure:
					bufferFactor === undefined ? { budget } : { budget, bufferFactor },
			},
		],
	};
}

async function prune(messages: ModelMessage[], pruningPolicy: PruningPolicy) {
	return new Prunella({ pruningPolicy }).prepare({
		messages,
		sessionId: "session",
		config: undefined,
	});
}

describe("olderThan messages", () => {
	test("counts the messages after a part's message", async () => {
		const { messages } = await prune(longTurn(3), {
			AND: [{ hasType: "tool-result" }, { olderThan: { messages: 1 } }],
		});

		expect(prunedCallIds(messages)).toEqual(["call-0", "call-1"]);
	});
});

describe("hasPressure", () => {
	test("prunes nothing while the conversation is within budget", async () => {
		const messages = longTurn(10);

		const result = await prune(messages, pressurePolicy(sizeOf(messages) + 1));

		expect(prunedCallIds(result.messages)).toEqual([]);
		expect(Object.keys(result.tools)).toEqual([]);
	});

	test("prunes the oldest matching parts until the conversation is back within budget", async () => {
		const messages = longTurn(10);
		const budget = Math.round(sizeOf(messages) * 0.7);

		const result = await prune(messages, pressurePolicy(budget, 0.2));

		const pruned = prunedCallIds(result.messages);
		expect(pruned.length).toBeGreaterThan(0);
		expect(pruned).toEqual(
			Array.from({ length: pruned.length }, (_, step) => `call-${step}`),
		);
		expect(sizeOf(result.messages)).toBeLessThanOrEqual(budget);
		expect(Object.keys(result.tools)).toEqual(["recall-pruned"]);
	});

	test("drops to about budget × (1 − bufferFactor) when the budget is first crossed", async () => {
		const messages = longTurn(10);
		const budget = sizeOf(messages) - 1;

		const result = await prune(messages, pressurePolicy(budget, 0.3));

		expect(sizeOf(result.messages)).toBeLessThanOrEqual(budget * 0.7 + 1);
	});

	test("keeps the same parts pruned while the conversation grows within a band", async () => {
		const messages = longTurn(10);
		const policy = pressurePolicy(sizeOf(messages) - 1, 0.3);
		const before = await prune(messages, policy);

		const grown: ModelMessage[] = [
			...messages,
			{ role: "assistant", content: [{ type: "text", text: "a short note" }] },
		];
		const after = await prune(grown, policy);

		expect(prunedCallIds(before.messages).length).toBeGreaterThan(0);
		expect(prunedCallIds(after.messages)).toEqual(
			prunedCallIds(before.messages),
		);
	});

	test("prunes more once the conversation grows into the next band", async () => {
		const messages = longTurn(10);
		const budget = Math.round(sizeOf(messages) * 0.5);
		const policy = pressurePolicy(budget, 0.1);
		const before = await prune(messages, policy);

		const after = await prune(longTurn(14), policy);

		expect(prunedCallIds(after.messages).length).toBeGreaterThan(
			prunedCallIds(before.messages).length,
		);
		expect(sizeOf(after.messages)).toBeLessThanOrEqual(budget);
	});

	test("defaults bufferFactor to 0.1", async () => {
		const messages = longTurn(10);
		const budget = Math.round(sizeOf(messages) * 0.6);

		const implicit = await prune(messages, pressurePolicy(budget));
		const explicit = await prune(messages, pressurePolicy(budget, 0.1));

		expect(prunedCallIds(implicit.messages)).toEqual(
			prunedCallIds(explicit.messages),
		);
	});

	test("prunes just enough to fit the budget when bufferFactor is 0", async () => {
		const messages = longTurn(10);
		const budget = Math.round(sizeOf(messages) * 0.6);

		const exact = await prune(messages, pressurePolicy(budget, 0));
		const buffered = await prune(messages, pressurePolicy(budget, 0.3));

		expect(sizeOf(exact.messages)).toBeLessThanOrEqual(budget);
		expect(prunedCallIds(exact.messages).length).toBeLessThan(
			prunedCallIds(buffered.messages).length,
		);
	});

	test("chooses only parts the rest of the policy allows", async () => {
		const messages = longTurn(10);

		const result = await prune(messages, pressurePolicy(0, 0));

		const pruned = prunedCallIds(result.messages);
		expect(pruned).not.toContain("call-9");
		const calls = result.messages.flatMap((message) =>
			message.role === "assistant" && Array.isArray(message.content)
				? message.content.filter((part) => part.type === "tool-call")
				: [],
		);
		expect(calls.every((call) => !("pruned" in (call.input as object)))).toBe(
			true,
		);
	});

	test("is false everywhere under NOT, so the rest of the policy decides", async () => {
		const messages = longTurn(4);

		const result = await prune(messages, {
			AND: [
				{ hasType: "tool-result" },
				{ NOT: { hasPressure: { budget: 0 } } },
			],
		});

		expect(prunedCallIds(result.messages)).toEqual([
			"call-0",
			"call-1",
			"call-2",
			"call-3",
		]);
	});

	test("rejects a bufferFactor outside [0, 1) and a negative budget", () => {
		for (const condition of [
			{ budget: 100, bufferFactor: 1 },
			{ budget: 100, bufferFactor: -0.1 },
			{ budget: -1 },
		]) {
			expect(
				() => new Prunella({ pruningPolicy: { hasPressure: condition } }),
			).toThrow();
		}
	});

	test("does not count parts that a summary already covers", async () => {
		const messages = longTurn(10);
		const parts = attachIdsToMessages(messages).flatMap(
			(message) => message.parts,
		);
		const summaries = new Map<string, CompactorSummary>([
			[
				"s1",
				{
					id: "s1",
					sessionId: "session",
					spans: [{ firstPartId: parts[1]!.id, lastPartId: parts[8]!.id }],
					text: "Steps 0 to 3.",
				},
			],
		]);
		const store: CompactorStore = {
			createSummary: async ({ summary }) => {
				summaries.set(summary.id, summary);
			},
			getSummary: async ({ id }) => summaries.get(id)!,
			getSummariesForSession: async () => [...summaries.values()],
			updateSummary: async ({ summary }) => {
				summaries.set(summary.id, summary);
			},
			deleteSummary: async ({ id }) => {
				summaries.delete(id);
			},
		};
		const withSummary = (pruningPolicy: PruningPolicy) =>
			new Prunella({
				pruningPolicy,
				compaction: {
					enabled: true,
					store,
					model: new MockLanguageModelV4(),
					policy: { compactionThreshold: Number.POSITIVE_INFINITY },
				},
			}).prepare({ messages, sessionId: "session", config: undefined });
		const unpruned = await withSummary({ hasRole: "never" as never });
		const budget = sizeOf(unpruned.messages) - 1;

		const result = await withSummary(pressurePolicy(budget, 0));

		expect(sizeOf(result.messages)).toBeLessThanOrEqual(budget);
		expect(prunedCallIds(result.messages)).toEqual(["call-4"]);
	});
});

import { describe, expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { Prunella } from "./index";
import type { PruningPolicy } from "./pruning";
import {
	CALLER_TOOLS,
	NO_PRUNING,
	createInMemoryStore,
	estimator,
	longTurn,
	makeMockModel,
	prune,
	prunedCallIds,
	sizeOf,
	userTexts,
} from "./test-utils";
import { attachIdsToMessages } from "./utils";

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
		expect(estimator.count(result)).toBeLessThanOrEqual(budget);
		expect(Object.keys(result.tools)).toEqual(["recall-pruned"]);
	});

	test("counts the tools that are sent with the messages", async () => {
		const messages = longTurn(10);
		const budget = sizeOf(messages) + 1;

		const withoutTools = await prune(messages, pressurePolicy(budget));
		const withTools = await prune(
			messages,
			pressurePolicy(budget),
			CALLER_TOOLS,
		);

		expect(prunedCallIds(withoutTools.messages)).toEqual([]);
		expect(prunedCallIds(withTools.messages).length).toBeGreaterThan(0);
		expect(
			estimator.count({
				messages: withTools.messages,
				tools: { ...CALLER_TOOLS, ...withTools.tools },
			}),
		).toBeLessThanOrEqual(budget);
		expect(Object.keys(withTools.tools)).toEqual(["recall-pruned"]);
	});

	test("counts the recall tool that pruning adds", async () => {
		const messages = longTurn(10);
		const budget = sizeOf(messages) - 1;

		const result = await prune(messages, pressurePolicy(budget, 0));

		const recallTool = estimator.count({ messages: [], tools: result.tools });
		expect(recallTool).toBeGreaterThan(estimator.count({ messages: [] }));
		expect(estimator.count(result)).toBeLessThanOrEqual(budget);
	});

	test("drops to about budget × (1 − bufferFactor) when the budget is first crossed", async () => {
		const messages = longTurn(10);
		const budget = sizeOf(messages) - 1;

		const result = await prune(messages, pressurePolicy(budget, 0.3));

		expect(estimator.count(result)).toBeLessThanOrEqual(budget * 0.7 + 1);
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
		expect(estimator.count(after)).toBeLessThanOrEqual(budget);
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

		expect(estimator.count(exact)).toBeLessThanOrEqual(budget);
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
				() =>
					new Prunella({
						estimator,
						pruningPolicy: { hasPressure: condition },
					}),
			).toThrow();
		}
	});

	test("does not count parts that a summary already covers", async () => {
		const messages = longTurn(10);
		const parts = attachIdsToMessages(messages).flatMap(
			(message) => message.parts,
		);
		const store = createInMemoryStore();
		store.summaries.set("s1", {
			id: "s1",
			sessionId: "session",
			spans: [{ firstPartId: parts[1]!.id, lastPartId: parts[8]!.id }],
			text: "Steps 0 to 3.",
		});
		const withSummary = (pruningPolicy: PruningPolicy) =>
			new Prunella({
				estimator,
				pruningPolicy,
				compaction: {
					enabled: true,
					store,
					model: makeMockModel(),
					policy: { compactionThreshold: Number.POSITIVE_INFINITY },
				},
			}).prepare({ messages, sessionId: "session", config: undefined });
		const unpruned = await withSummary(NO_PRUNING);
		const budget = sizeOf(unpruned.messages) - 1;

		const result = await withSummary(pressurePolicy(budget, 0));

		expect(estimator.count(result)).toBeLessThanOrEqual(budget);
		expect(prunedCallIds(result.messages)).toEqual(["call-4"]);
	});
});

describe("pressure after compaction", () => {
	test("chooses parts again once a summary covers what was pruned", async () => {
		const messages = longTurn(10);
		// Room for the user message, the last step, and a summary, but not for nine placeholders.
		const budget = sizeOf([messages[0]!, ...messages.slice(-2)]) + 100;

		const result = await new Prunella({
			estimator,
			pruningPolicy: {
				AND: [
					{ OR: [{ hasType: "tool-call" }, { hasType: "tool-result" }] },
					{ olderThan: { messages: 1 } },
					{ hasPressure: { budget } },
				],
			},
			compaction: {
				enabled: true,
				store: createInMemoryStore(),
				model: makeMockModel(),
				policy: {
					compactionThreshold: budget,
					minCompactableSpan: 0,
					keepRecent: { tokens: Math.round(budget / 4) },
				},
			},
		}).prepare({ messages, sessionId: "session", config: undefined });

		expect(userTexts(result.messages)[1]).toContain("<Summary");
		expect(estimator.count(result)).toBeLessThanOrEqual(budget);
		expect(prunedCallIds(result.messages)).toEqual([]);
		expect(Object.keys(result.tools)).toEqual([]);
	});
});

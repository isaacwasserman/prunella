import { describe, expect, test } from "bun:test";
import { longTurn, partsSizeOfLast, prune, prunedCallIds } from "./test-utils";

describe("olderThan messages", () => {
	test("counts the messages after a part's message", async () => {
		const { messages } = await prune(longTurn(3), {
			AND: [{ hasType: "tool-result" }, { olderThan: { messages: 1 } }],
		});

		expect(prunedCallIds(messages)).toEqual(["call-0", "call-1"]);
	});
});

describe("olderThan tokens", () => {
	test("counts the original tokens of the parts after a part", async () => {
		const messages = longTurn(4);
		const lastStep = partsSizeOfLast(messages, 2);

		const { messages: pruned } = await prune(messages, {
			AND: [{ hasType: "tool-result" }, { olderThan: { tokens: lastStep } }],
		});

		expect(prunedCallIds(pruned)).toEqual(["call-0", "call-1"]);
	});

	test("prunes a part once more than that many tokens follow it", async () => {
		const messages = longTurn(4);
		const lastStep = partsSizeOfLast(messages, 2);

		const { messages: pruned } = await prune(messages, {
			AND: [
				{ hasType: "tool-result" },
				{ olderThan: { tokens: lastStep - 1 } },
			],
		});

		expect(prunedCallIds(pruned)).toEqual(["call-0", "call-1", "call-2"]);
	});
});

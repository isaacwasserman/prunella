import { describe, expect, test } from "bun:test";
import { longTurn, prune, prunedCallIds } from "./test-utils";

describe("olderThan messages", () => {
	test("counts the messages after a part's message", async () => {
		const { messages } = await prune(longTurn(3), {
			AND: [{ hasType: "tool-result" }, { olderThan: { messages: 1 } }],
		});

		expect(prunedCallIds(messages)).toEqual(["call-0", "call-1"]);
	});
});

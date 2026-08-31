import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { Prunella } from "./index";

const PLACEHOLDER_PREFIX = "This part of the message has been pruned";

function baseMessages(): ModelMessage[] {
	return [
		{ role: "user", content: "hello" },
		{ role: "assistant", content: [{ type: "text", text: "hi there" }] },
		{ role: "user", content: "second question" },
	];
}

test("prunes parts that match the policy and leaves others intact", () => {
	const prunella = new Prunella(
		baseMessages(),
		{ hasRole: "assistant" },
		{ recallDuration: { turns: 2 } },
	);

	const { messages, tools } = prunella.prepare();

	// The assistant text part is replaced by a placeholder.
	const assistantContent = messages[1]?.content;
	if (!Array.isArray(assistantContent))
		throw new Error("expected array content");
	const assistantPart = assistantContent[0];
	if (typeof assistantPart === "string" || assistantPart?.type !== "text") {
		throw new Error("expected a text part");
	}
	expect(assistantPart.text.startsWith(PLACEHOLDER_PREFIX)).toBe(true);

	// The user string content is untouched.
	expect(messages[0]?.content).toBe("hello");

	// The recall tool is exposed to the model.
	expect(tools).toHaveProperty("recall-pruned");
});

test("a recall request restores the pruned part", () => {
	const messages = baseMessages();
	const recallPolicy = { recallDuration: { turns: 2 } } as const;

	// First pass: prune, then read the pruneId out of the placeholder.
	const firstPass = new Prunella(
		messages,
		{ hasRole: "assistant" },
		recallPolicy,
	).prepare();
	const prunedContent = firstPass.messages[1]?.content;
	if (!Array.isArray(prunedContent) || prunedContent[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	const pruneId = prunedContent[0].text.match(/pruneId "([a-f0-9]+)"/)?.[1];
	expect(pruneId).toBeString();

	// Second pass: append a recall request for that pruneId.
	const withRecall: ModelMessage[] = [
		...messages,
		{
			role: "assistant",
			content: [
				{
					type: "tool-call",
					toolCallId: "call-1",
					toolName: "recall-pruned",
					input: { pruneId },
				},
			],
		},
	];
	const secondPass = new Prunella(
		withRecall,
		{ hasRole: "assistant" },
		recallPolicy,
	).prepare();

	const restoredContent = secondPass.messages[1]?.content;
	if (!Array.isArray(restoredContent) || restoredContent[0]?.type !== "text") {
		throw new Error("expected a restored text part");
	}
	expect(restoredContent[0].text).toBe("hi there");
});

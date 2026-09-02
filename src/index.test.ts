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

test("prunes parts that match the policy and leaves others intact", async () => {
	const prunella = new Prunella({
		pruningPolicy: { hasRole: "assistant" },
	});

	const { messages, tools } = await prunella.prepare({
		messages: baseMessages(),
		sessionId: "test",
		config: undefined,
	});

	const assistantContent = messages[1]?.content;
	if (!Array.isArray(assistantContent))
		throw new Error("expected array content");
	const assistantPart = assistantContent[0];
	if (typeof assistantPart === "string" || assistantPart?.type !== "text") {
		throw new Error("expected a text part");
	}
	expect(assistantPart.text.startsWith(PLACEHOLDER_PREFIX)).toBe(true);

	const userContent = messages[0]?.content;
	if (!Array.isArray(userContent)) throw new Error("expected array content");
	expect(userContent[0]).toEqual({ type: "text", text: "hello" });

	expect(tools).toHaveProperty("recall-pruned");
});

test("placeholder is permanent after recall request", async () => {
	const messages = baseMessages();

	const firstPass = await new Prunella({
		pruningPolicy: { hasRole: "assistant" },
	}).prepare({ messages, sessionId: "test", config: undefined });

	const prunedContent = firstPass.messages[1]?.content;
	if (!Array.isArray(prunedContent) || prunedContent[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	const pruneId = prunedContent[0].text.match(/pruneId "([a-f0-9]+)"/)?.[1];
	expect(pruneId).toBeString();

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

	const secondPass = await new Prunella({
		pruningPolicy: { hasRole: "assistant" },
	}).prepare({ messages: withRecall, sessionId: "test", config: undefined });

	const stillPruned = secondPass.messages[1]?.content;
	if (!Array.isArray(stillPruned) || stillPruned[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	expect(stillPruned[0].text.startsWith(PLACEHOLDER_PREFIX)).toBe(true);
});

test("recall tool returns original content", async () => {
	const { tools, messages } = await new Prunella({
		pruningPolicy: { hasRole: "assistant" },
	}).prepare({
		messages: baseMessages(),
		sessionId: "test",
		config: undefined,
	});

	const prunedContent = messages[1]?.content;
	if (!Array.isArray(prunedContent) || prunedContent[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	const pruneId = prunedContent[0].text.match(/pruneId "([a-f0-9]+)"/)?.[1];
	expect(pruneId).toBeString();

	const recallTool = tools["recall-pruned"];
	const result = await recallTool.execute({ pruneId }, {
		toolCallId: "call-1",
		messages: [],
	} as unknown as Parameters<typeof recallTool.execute>[1]);
	expect(result).toBe("hi there");
});

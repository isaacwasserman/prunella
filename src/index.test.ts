import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { Prunella } from "./index";
import { PLACEHOLDER_PREFIX, estimator } from "./test-utils";

function baseMessages(): ModelMessage[] {
	return [
		{ role: "user", content: "hello" },
		{ role: "assistant", content: [{ type: "text", text: "hi there" }] },
		{ role: "user", content: "second question" },
	];
}

test("prunes parts that match the policy and leaves others intact", async () => {
	const prunella = new Prunella({
		estimator,
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

	expect(messages[0]).toEqual({ role: "user", content: "hello" });
	expect(tools).toEqual({});
});

test("placeholder is permanent after recall request", async () => {
	const messages = baseMessages();

	const firstPass = await new Prunella({
		estimator,
		pruningPolicy: { hasRole: "assistant" },
	}).prepare({ messages, sessionId: "test", config: undefined });

	const prunedContent = firstPass.messages[1]?.content;
	if (!Array.isArray(prunedContent) || prunedContent[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	const pruneId = prunedContent[0].text.match(
		/pruneId "([A-Za-z0-9_-]{8})"/,
	)?.[1];
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
		estimator,
		pruningPolicy: { hasRole: "assistant" },
	}).prepare({ messages: withRecall, sessionId: "test", config: undefined });

	const stillPruned = secondPass.messages[1]?.content;
	if (!Array.isArray(stillPruned) || stillPruned[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	expect(stillPruned[0].text.startsWith(PLACEHOLDER_PREFIX)).toBe(true);
});

test("recall tool returns original content", async () => {
	const prunella = new Prunella({
		estimator,
		pruningPolicy: { hasRole: "assistant" },
	});
	const { messages } = await prunella.prepare({
		messages: baseMessages(),
		sessionId: "test",
		config: undefined,
	});

	const prunedContent = messages[1]?.content;
	if (!Array.isArray(prunedContent) || prunedContent[0]?.type !== "text") {
		throw new Error("expected a pruned text part");
	}
	const pruneId = prunedContent[0].text.match(
		/pruneId "([A-Za-z0-9_-]{8})"/,
	)?.[1];
	expect(pruneId).toBeString();

	const recallTool = prunella.tools["recall-pruned"];
	const result = await recallTool.execute({ pruneId }, {
		toolCallId: "call-1",
		messages: [],
	} as unknown as Parameters<typeof recallTool.execute>[1]);
	expect(result).toBe("hi there");
});

test("one recall tool recalls parts from each prepare call", async () => {
	const prunella = new Prunella({
		estimator,
		pruningPolicy: { hasRole: "assistant" },
	});
	const recallTool = prunella.tools["recall-pruned"];
	const pruneIdOf = async (reply: string, sessionId: string) => {
		const { messages } = await prunella.prepare({
			messages: [
				{ role: "user", content: "hello" },
				{ role: "assistant", content: [{ type: "text", text: reply }] },
			],
			sessionId,
			config: undefined,
		});
		const content = messages[1]?.content;
		if (!Array.isArray(content) || content[0]?.type !== "text") {
			throw new Error("expected a pruned text part");
		}
		return content[0].text.match(/pruneId "([A-Za-z0-9_-]{8})"/)?.[1];
	};

	const first = await pruneIdOf("first reply", "session-1");
	const second = await pruneIdOf("second reply", "session-2");

	expect(prunella.tools["recall-pruned"]).toBe(recallTool);
	const options = {
		toolCallId: "call-1",
		messages: [],
	} as unknown as Parameters<typeof recallTool.execute>[1];
	expect(await recallTool.execute({ pruneId: first }, options)).toBe(
		"first reply",
	);
	expect(await recallTool.execute({ pruneId: second }, options)).toBe(
		"second reply",
	);
});

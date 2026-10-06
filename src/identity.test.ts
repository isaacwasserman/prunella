import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { attachIdsToMessages } from "./utils";

const conversation: ModelMessage[] = [
	{ role: "user", content: "Look it up" },
	{
		role: "assistant",
		content: [
			{ type: "reasoning", text: "I will call lookup." },
			{
				type: "tool-call",
				toolCallId: "call-1",
				toolName: "lookup",
				input: { query: "x" },
			},
		],
	},
	{
		role: "tool",
		content: [
			{
				type: "tool-result",
				toolCallId: "call-1",
				toolName: "lookup",
				output: { type: "json", value: { rows: 3 } },
			},
		],
	},
	{ role: "assistant", content: [{ type: "text", text: "Three rows." }] },
];

function ids(messages: ModelMessage[]): string[] {
	return attachIdsToMessages(messages).flatMap((message) =>
		message.parts.map((part) => part.id),
	);
}

test("part IDs ignore provider fields and how a tool's input and output are written", () => {
	const stored: ModelMessage[] = [
		{
			role: "user",
			content: [{ type: "text", text: "Look it up" }],
			providerOptions: { bedrock: { cachePoint: { type: "default" } } },
		},
		{
			role: "assistant",
			content: [
				{
					type: "reasoning",
					text: "I will call lookup.",
					providerOptions: { bedrock: { signature: "abc" } },
				},
				{
					type: "tool-call",
					toolCallId: "call-1",
					toolName: "lookup",
					input: '{"query":"x"}',
				},
			],
		},
		{
			role: "tool",
			content: [
				{
					type: "tool-result",
					toolCallId: "call-1",
					toolName: "lookup",
					output: { type: "text", value: '{"rows":3}' },
				},
			],
		},
		{ role: "assistant", content: "Three rows." },
	];

	expect(ids(stored)).toEqual(ids(conversation));
});

test("part IDs change with the content of a part", () => {
	const changed = conversation.with(3, {
		role: "assistant",
		content: [{ type: "text", text: "Four rows." }],
	});

	expect(ids(changed).slice(0, 4)).toEqual(ids(conversation).slice(0, 4));
	expect(ids(changed)[4]).not.toBe(ids(conversation)[4]);
});

test("the same text after different messages gets different IDs", () => {
	const repeated: ModelMessage[] = [
		{ role: "user", content: "first" },
		{ role: "assistant", content: "ok" },
		{ role: "user", content: "second" },
		{ role: "assistant", content: "ok" },
	];

	expect(new Set(ids(repeated)).size).toBe(4);
});

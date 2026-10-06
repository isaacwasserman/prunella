import type { LanguageModelV4GenerateResult } from "@ai-sdk/provider";
import type { ModelMessage } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type { CompactorStore, CompactorSummary } from "./compaction";
import { Prunella } from "./index";
import type { PruningPolicy } from "./pruning";
import { attachIdsToMessages, partTokens } from "./utils";

export const PLACEHOLDER_PREFIX = "This part of the message has been pruned";

/** A pruning policy that prunes nothing. */
export const NO_PRUNING: PruningPolicy = { hasRole: "never-matches" as never };

const MOCK_USAGE: LanguageModelV4GenerateResult["usage"] = {
	inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
	outputTokens: { total: 10, text: 10, reasoning: 0 },
};

function makeMockResult(summary: string): LanguageModelV4GenerateResult {
	return {
		content: [{ type: "text", text: JSON.stringify({ summary }) }],
		finishReason: { unified: "stop", raw: "stop" },
		usage: MOCK_USAGE,
		warnings: [],
	};
}

export function makeMockModel(id = "mock") {
	return new MockLanguageModelV4({
		modelId: id,
		doGenerate: async ({ prompt }) => {
			const text = typeof prompt === "string" ? prompt : JSON.stringify(prompt);
			const first = text.slice(0, 20);
			const last = text.slice(-20);
			return makeMockResult(`${first}...${last}`);
		},
	});
}

export function createInMemoryStore(): CompactorStore & {
	summaries: Map<string, CompactorSummary>;
} {
	const summaries = new Map<string, CompactorSummary>();
	return {
		summaries,
		createSummary: async ({ summary }) => {
			summaries.set(summary.id, summary);
		},
		getSummary: async ({ id }) => {
			const s = summaries.get(id);
			if (!s) throw new Error(`Summary ${id} not found`);
			return s;
		},
		getSummariesForSession: async ({ sessionId }) => {
			return [...summaries.values()].filter((s) => s.sessionId === sessionId);
		},
		updateSummary: async ({ summary }) => {
			summaries.set(summary.id, summary);
		},
		deleteSummary: async ({ id }) => {
			summaries.delete(id);
		},
	};
}

/** One user turn with `steps` tool calls, each returning a large result. */
export function longTurn(steps: number): ModelMessage[] {
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

/** Tokens of the parts in the last `count` messages, as a tokens part age counts them. */
export function partsSizeOfLast(
	messages: ModelMessage[],
	count: number,
): number {
	return attachIdsToMessages(messages)
		.slice(-count)
		.flatMap((message) => message.parts)
		.reduce((total, part) => total + partTokens(part), 0);
}

/** The call IDs of the tool results that were pruned. */
export function prunedCallIds(messages: ModelMessage[]): string[] {
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

/** The text of every user message, summaries included. */
export function userTexts(messages: ModelMessage[]): string[] {
	return messages.flatMap((message) => {
		if (message.role !== "user") return [];
		if (typeof message.content === "string") return [message.content];
		return message.content.flatMap((part) =>
			part.type === "text" ? [part.text] : [],
		);
	});
}

/** The call IDs of the tool results that were sent. */
export function resultIds(messages: ModelMessage[]): string[] {
	return messages.flatMap((message) =>
		message.role === "tool"
			? message.content.flatMap((part) =>
					part.type === "tool-result" ? [part.toolCallId] : [],
				)
			: [],
	);
}

export async function prune(
	messages: ModelMessage[],
	pruningPolicy: PruningPolicy,
) {
	return new Prunella({ pruningPolicy }).prepare({
		messages,
		sessionId: "session",
		config: undefined,
	});
}

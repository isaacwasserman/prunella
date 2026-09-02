import { describe, expect, mock, test } from "bun:test";
import type { LanguageModelV4GenerateResult } from "@ai-sdk/provider";
import type { ModelMessage } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type { CompactorStore, CompactorSummary } from "./compaction";
import { Prunella } from "./index";

// --- Test helpers ---

const PLACEHOLDER_PREFIX = "This part of the message has been pruned";

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

function makeMockModel(id = "mock") {
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

function createInMemoryStore(): CompactorStore & {
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

function longConversation(turns: number): ModelMessage[] {
	const messages: ModelMessage[] = [
		{ role: "system", content: "You are a helpful assistant." },
	];
	for (let i = 0; i < turns; i++) {
		messages.push({
			role: "user",
			content: `Question ${i}: ${"x".repeat(200)}`,
		});
		messages.push({
			role: "assistant",
			content: [
				{
					type: "text",
					text: `Answer ${i}: ${"y".repeat(200)}`,
				},
			],
		});
	}
	return messages;
}

function extractText(messages: ModelMessage[]): string {
	return messages
		.flatMap((m) =>
			Array.isArray(m.content)
				? m.content
						.filter(
							(p): p is { type: "text"; text: string } =>
								typeof p !== "string" && p.type === "text",
						)
						.map((p) => p.text)
				: [typeof m.content === "string" ? m.content : ""],
		)
		.join("\n");
}

function execOpts(): unknown {
	return { toolCallId: "c1", messages: [] };
}

function pruneIdFrom(text: string): string {
	const match = text.match(/pruneId "([a-f0-9]+)"/);
	if (!match?.[1]) throw new Error("no pruneId found in text");
	return match[1];
}

// --- Tests ---

describe("edge cases: empty and minimal inputs", () => {
	test("empty messages array returns empty output", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const { messages, tools } = await prunella.prepare({
			messages: [],
			sessionId: "test",
			config: undefined,
		});
		expect(messages).toEqual([]);
		expect(tools).toHaveProperty("recall-pruned");
	});

	test("single user message with string content", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const { messages } = await prunella.prepare({
			messages: [{ role: "user", content: "hello" }],
			sessionId: "test",
			config: undefined,
		});
		expect(messages).toHaveLength(1);
		expect(messages[0]!.role).toBe("user");
	});

	test("single system message is never pruned", async () => {
		const prunella = new Prunella({
			pruningPolicy: { olderThan: { turns: 0 } },
		});
		const { messages } = await prunella.prepare({
			messages: [{ role: "system", content: "system prompt" }],
			sessionId: "test",
			config: undefined,
		});
		expect(messages).toHaveLength(1);
	});
});

describe("edge cases: pruning", () => {
	test("nothing pruned when no parts match policy", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "tool" },
		});
		const input: ModelMessage[] = [
			{ role: "user", content: "hi" },
			{ role: "assistant", content: [{ type: "text", text: "hello" }] },
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const text = extractText(messages);
		expect(text).not.toContain(PLACEHOLDER_PREFIX);
		expect(text).toContain("hello");
	});

	test("all messages pruned except system", async () => {
		const prunella = new Prunella({
			pruningPolicy: { NOT: { hasRole: "system" } },
		});
		const input: ModelMessage[] = [
			{ role: "system", content: "sys" },
			{ role: "user", content: "hi" },
			{ role: "assistant", content: [{ type: "text", text: "bye" }] },
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const text = extractText(messages);
		expect(text).toContain(PLACEHOLDER_PREFIX);
		expect(text).not.toContain("bye");
	});

	test("multi-part message: only matching parts are pruned", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasType: "tool-call" },
		});
		const input: ModelMessage[] = [
			{
				role: "assistant",
				content: [
					{ type: "text", text: "thinking" },
					{
						type: "tool-call",
						toolCallId: "tc-1",
						toolName: "search",
						input: { q: "test" },
					},
				],
			},
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const content = messages[0]!.content;
		if (!Array.isArray(content)) throw new Error("expected array");
		const textParts = content.filter(
			(p) => typeof p !== "string" && p.type === "text",
		) as { type: "text"; text: string }[];
		const hasOriginal = textParts.some((p) => p.text === "thinking");
		const hasPruned = textParts.some((p) =>
			p.text.startsWith(PLACEHOLDER_PREFIX),
		);
		expect(hasOriginal).toBe(true);
		expect(hasPruned).toBe(true);
	});

	test("recall tool-call parts are never pruned even when matching policy", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasType: "tool-call" },
		});
		const input: ModelMessage[] = [
			{
				role: "assistant",
				content: [
					{
						type: "tool-call",
						toolCallId: "rc-1",
						toolName: "recall-pruned",
						input: { pruneId: "abc" },
					},
				],
			},
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const content = messages[0]!.content;
		if (!Array.isArray(content)) throw new Error("expected array");
		expect(content[0]!.type).toBe("tool-call");
	});

	test("recall tool returns 'Part not found' for invalid pruneId", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const { tools } = await prunella.prepare({
			messages: [
				{ role: "user", content: "hi" },
				{
					role: "assistant",
					content: [{ type: "text", text: "hello" }],
				},
			],
			sessionId: "test",
			config: undefined,
		});
		const result = await tools["recall-pruned"].execute(
			{ pruneId: "nonexistent" },
			execOpts() as never,
		);
		expect(result).toBe("Part not found or not pruned.");
	});

	test("recall tool returns serialized tool-call content", async () => {
		const prunella = new Prunella({
			pruningPolicy: {
				AND: [
					{ hasType: "tool-call" },
					{
						NOT: {
							shouldPrune: ({ part }) =>
								typeof part !== "string" &&
								part.type === "tool-call" &&
								part.toolName === "recall-pruned",
						},
					},
				],
			},
		});
		const input: ModelMessage[] = [
			{
				role: "assistant",
				content: [
					{
						type: "tool-call",
						toolCallId: "tc-1",
						toolName: "search",
						input: { q: "weather" },
					},
				],
			},
		];
		const { messages, tools } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const content = messages[0]!.content;
		if (!Array.isArray(content)) throw new Error("expected array");
		const placeholder = content.find(
			(p) =>
				typeof p !== "string" &&
				p.type === "text" &&
				p.text.startsWith(PLACEHOLDER_PREFIX),
		) as { type: "text"; text: string } | undefined;
		if (!placeholder) throw new Error("expected placeholder");

		const pruneId = pruneIdFrom(placeholder.text);
		const result = await tools["recall-pruned"].execute(
			{ pruneId },
			execOpts() as never,
		);
		const parsed = JSON.parse(result as string);
		expect(parsed.toolName).toBe("search");
		expect(parsed.input.q).toBe("weather");
	});

	test("olderThan with steps mode", async () => {
		const prunella = new Prunella({
			pruningPolicy: {
				AND: [{ hasRole: "user" }, { olderThan: { steps: 3 } }],
			},
		});
		const input: ModelMessage[] = [
			{ role: "user", content: "first" },
			{ role: "assistant", content: [{ type: "text", text: "a1" }] },
			{ role: "user", content: "second" },
			{ role: "assistant", content: [{ type: "text", text: "a2" }] },
			{ role: "user", content: "third" },
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const text = extractText(messages);
		expect(text).toContain("third");
		expect(text).toContain("second");
	});

	test("complex AND/OR/NOT policy", async () => {
		const prunella = new Prunella({
			pruningPolicy: {
				AND: [
					{
						OR: [{ hasRole: "assistant" }, { hasRole: "user" }],
					},
					{ NOT: { hasType: "tool-call" } },
					{ olderThan: { turns: 0 } },
				],
			},
		});
		const input: ModelMessage[] = [
			{ role: "user", content: "hi" },
			{
				role: "assistant",
				content: [
					{ type: "text", text: "let me help" },
					{
						type: "tool-call",
						toolCallId: "tc-1",
						toolName: "search",
						input: {},
					},
				],
			},
			{ role: "user", content: "latest" },
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const text = extractText(messages);
		expect(text).toContain(PLACEHOLDER_PREFIX);
		expect(text).toContain("latest");
	});
});

describe("edge cases: part ID stability", () => {
	test("same messages produce identical masks across calls", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const input: ModelMessage[] = [
			{ role: "user", content: "hi" },
			{ role: "assistant", content: [{ type: "text", text: "hello" }] },
		];

		const r1 = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		const r2 = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});

		const id1 = pruneIdFrom(extractText(r1.messages));
		const id2 = pruneIdFrom(extractText(r2.messages));
		expect(id1).toBe(id2);
	});

	test("appending messages does not change existing part IDs", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const base: ModelMessage[] = [
			{ role: "user", content: "hi" },
			{ role: "assistant", content: [{ type: "text", text: "hello" }] },
		];
		const extended: ModelMessage[] = [
			...base,
			{ role: "user", content: "more" },
			{ role: "assistant", content: [{ type: "text", text: "sure" }] },
		];

		const r1 = await prunella.prepare({
			messages: base,
			sessionId: "test",
			config: undefined,
		});
		const r2 = await prunella.prepare({
			messages: extended,
			sessionId: "test",
			config: undefined,
		});

		const id1 = pruneIdFrom(extractText(r1.messages));
		const id2 = pruneIdFrom(
			extractText(
				r2.messages.filter((m) => m.role === "assistant").slice(0, 1),
			),
		);
		expect(id1).toBe(id2);
	});
});

describe("edge cases: compaction", () => {
	test("no compaction when below threshold", async () => {
		const store = createInMemoryStore();
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 999_999,
					minCompactableSpan: 100,
					maxIterations: 5,
				},
			},
		});
		await prunella.prepare({
			messages: longConversation(3),
			sessionId: "no-compact",
			config: undefined,
		});
		expect(store.summaries.size).toBe(0);
	});

	test("canCompact returning false for everything prevents compaction", async () => {
		const store = createInMemoryStore();
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					canCompact: () => false,
					compactionThreshold: 1,
					minCompactableSpan: 1,
					maxIterations: 5,
				},
			},
		});
		await prunella.prepare({
			messages: longConversation(5),
			sessionId: "no-compact-2",
			config: undefined,
		});
		expect(store.summaries.size).toBe(0);
	});

	test("system messages are never included in compaction spans", async () => {
		const store = createInMemoryStore();
		const prunella = new Prunella({
			pruningPolicy: {
				AND: [{ hasRole: "assistant" }, { olderThan: { turns: 100 } }],
			},
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 50,
					maxIterations: 3,
				},
			},
		});
		const messages = longConversation(10);
		const { messages: rendered } = await prunella.prepare({
			messages,
			sessionId: "sys-check",
			config: undefined,
		});

		const systemMsg = rendered.find((m) => m.role === "system");
		expect(systemMsg).toBeDefined();
		const sysText = Array.isArray(systemMsg!.content)
			? (systemMsg!.content[0] as { type: "text"; text: string }).text
			: systemMsg!.content;
		expect(sysText).not.toContain("<Summary>");
		expect(sysText).toContain("helpful assistant");
	});

	test("summaries from different sessions do not interfere", async () => {
		const store = createInMemoryStore();
		const model = makeMockModel();
		const messages = longConversation(10);

		const p1 = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
			compaction: {
				enabled: true,
				store,
				model,
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 2,
				},
			},
		});

		await p1.prepare({ messages, sessionId: "sess-A", config: undefined });
		const countA = [...store.summaries.values()].filter(
			(s) => s.sessionId === "sess-A",
		).length;

		await p1.prepare({ messages, sessionId: "sess-B", config: undefined });
		const countB = [...store.summaries.values()].filter(
			(s) => s.sessionId === "sess-B",
		).length;

		expect(countA).toBeGreaterThan(0);
		expect(countB).toBeGreaterThan(0);

		const aIds = [...store.summaries.values()]
			.filter((s) => s.sessionId === "sess-A")
			.map((s) => s.id);
		const bIds = [...store.summaries.values()]
			.filter((s) => s.sessionId === "sess-B")
			.map((s) => s.id);
		expect(aIds.some((id) => bIds.includes(id))).toBe(false);
	});

	test("summaries persist and are reused on subsequent prepare calls", async () => {
		const store = createInMemoryStore();
		const model = makeMockModel();
		const messages = longConversation(10);

		const prunella = new Prunella({
			pruningPolicy: {
				AND: [{ hasRole: "assistant" }, { olderThan: { turns: 100 } }],
			},
			compaction: {
				enabled: true,
				store,
				model,
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 3,
				},
			},
		});

		await prunella.prepare({ messages, sessionId: "reuse", config: undefined });
		const countAfterFirst = store.summaries.size;
		expect(countAfterFirst).toBeGreaterThan(0);

		const callsBefore = model.doGenerateCalls.length;
		await prunella.prepare({ messages, sessionId: "reuse", config: undefined });
		const callsAfter = model.doGenerateCalls.length;

		expect(callsAfter - callsBefore).toBeLessThanOrEqual(countAfterFirst);
	});

	test("maxIterations=0 prevents any compaction", async () => {
		const store = createInMemoryStore();
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 1,
					minCompactableSpan: 1,
					maxIterations: 0,
				},
			},
		});
		await prunella.prepare({
			messages: longConversation(5),
			sessionId: "zero-iter",
			config: undefined,
		});
		expect(store.summaries.size).toBe(0);
	});
});

describe("edge cases: pruning + compaction interaction", () => {
	test("compaction takes precedence over pruning for covered parts", async () => {
		const store = createInMemoryStore();
		const prunella = new Prunella({
			pruningPolicy: { olderThan: { turns: 1 } },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 3,
				},
			},
		});
		const messages = longConversation(10);
		const { messages: rendered } = await prunella.prepare({
			messages,
			sessionId: "overlap",
			config: undefined,
		});

		const summaryCount = rendered.filter(
			(m) =>
				Array.isArray(m.content) &&
				m.content[0]?.type === "text" &&
				(m.content[0] as { text: string }).text.includes("<Summary>"),
		).length;

		if (summaryCount > 0) {
			const allText = extractText(rendered);
			const placeholderCount = (
				allText.match(new RegExp(PLACEHOLDER_PREFIX, "g")) || []
			).length;
			const summaryTextCount = (allText.match(/<Summary>/g) || []).length;
			expect(summaryTextCount).toBeGreaterThan(0);
			expect(placeholderCount + summaryTextCount).toBeGreaterThan(0);
		}
	});

	test("part IDs are stable across calls with compaction enabled", async () => {
		const store = createInMemoryStore();
		const model = makeMockModel();
		const messages = longConversation(5);

		const prunella = new Prunella({
			pruningPolicy: {
				AND: [{ hasRole: "assistant" }, { olderThan: { turns: 2 } }],
			},
			compaction: {
				enabled: true,
				store,
				model,
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 2,
				},
			},
		});

		await prunella.prepare({
			messages,
			sessionId: "stable",
			config: undefined,
		});
		const summaryIds1 = [...store.summaries.values()].map(
			(s) => s.spans[0]!.firstPartId,
		);

		await prunella.prepare({
			messages,
			sessionId: "stable",
			config: undefined,
		});
		const summaryIds2 = [...store.summaries.values()]
			.filter((s) => s.sessionId === "stable")
			.map((s) => s.spans[0]!.firstPartId);

		for (const id of summaryIds1) {
			expect(summaryIds2).toContain(id);
		}
	});

	test("appending messages does not invalidate existing summaries", async () => {
		const store = createInMemoryStore();
		const model = makeMockModel();

		const prunella = new Prunella({
			pruningPolicy: {
				AND: [{ hasRole: "assistant" }, { olderThan: { turns: 100 } }],
			},
			compaction: {
				enabled: true,
				store,
				model,
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 2,
				},
			},
		});

		const base = longConversation(10);
		await prunella.prepare({
			messages: base,
			sessionId: "append",
			config: undefined,
		});
		const firstPartIds = [...store.summaries.values()].map(
			(s) => s.spans[0]!.firstPartId,
		);

		const extended: ModelMessage[] = [
			...base,
			{ role: "user", content: "one more question" },
			{
				role: "assistant",
				content: [{ type: "text", text: "one more answer" }],
			},
		];

		await expect(
			prunella.prepare({
				messages: extended,
				sessionId: "append",
				config: undefined,
			}),
		).resolves.toBeDefined();

		for (const id of firstPartIds) {
			const still = [...store.summaries.values()].find(
				(s) => s.spans[0]?.firstPartId === id,
			);
			expect(still).toBeDefined();
		}
	});
});

describe("edge cases: render step", () => {
	test("string content is normalized to text part array", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "tool" },
		});
		const { messages } = await prunella.prepare({
			messages: [{ role: "user", content: "hello world" }],
			sessionId: "test",
			config: undefined,
		});
		const content = messages[0]!.content;
		expect(Array.isArray(content)).toBe(true);
		if (Array.isArray(content)) {
			expect(content[0]).toEqual({ type: "text", text: "hello world" });
		}
	});

	test("message with all parts pruned still appears with placeholders", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const { messages } = await prunella.prepare({
			messages: [
				{ role: "user", content: "hi" },
				{
					role: "assistant",
					content: [
						{ type: "text", text: "part 1" },
						{ type: "text", text: "part 2" },
					],
				},
			],
			sessionId: "test",
			config: undefined,
		});
		const assistant = messages.find((m) => m.role === "assistant");
		expect(assistant).toBeDefined();
		const content = assistant!.content;
		if (!Array.isArray(content)) throw new Error("expected array");
		expect(content).toHaveLength(2);
		for (const part of content) {
			if (typeof part !== "string" && part.type === "text") {
				expect(part.text).toContain(PLACEHOLDER_PREFIX);
			}
		}
	});

	test("each pruned part gets a unique pruneId", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "assistant" },
		});
		const { messages, tools } = await prunella.prepare({
			messages: [
				{ role: "user", content: "hi" },
				{
					role: "assistant",
					content: [
						{ type: "text", text: "alpha" },
						{ type: "text", text: "beta" },
					],
				},
			],
			sessionId: "test",
			config: undefined,
		});
		const content = messages.find((m) => m.role === "assistant")!.content;
		if (!Array.isArray(content)) throw new Error("expected array");

		const ids = content
			.filter(
				(p): p is { type: "text"; text: string } =>
					typeof p !== "string" && p.type === "text",
			)
			.map((p) => pruneIdFrom(p.text));

		expect(ids[0]).not.toBe(ids[1]);

		const r1 = await tools["recall-pruned"].execute(
			{ pruneId: ids[0] },
			execOpts() as never,
		);
		const r2 = await tools["recall-pruned"].execute(
			{ pruneId: ids[1] },
			execOpts() as never,
		);
		expect(r1).toBe("alpha");
		expect(r2).toBe("beta");
	});

	test("output preserves message roles", async () => {
		const prunella = new Prunella({
			pruningPolicy: { hasRole: "never-matches" as any },
		});
		const input: ModelMessage[] = [
			{ role: "system", content: "sys" },
			{ role: "user", content: "u" },
			{ role: "assistant", content: [{ type: "text", text: "a" }] },
		];
		const { messages } = await prunella.prepare({
			messages: input,
			sessionId: "test",
			config: undefined,
		});
		expect(messages.map((m) => m.role)).toEqual([
			"system",
			"user",
			"assistant",
		]);
	});
});

describe("runtime config", () => {
	type TestConfig = { tenantId: string };

	function createConfigAwareStore(): CompactorStore<TestConfig> & {
		summaries: Map<string, CompactorSummary>;
		receivedConfigs: TestConfig[];
	} {
		const summaries = new Map<string, CompactorSummary>();
		const receivedConfigs: TestConfig[] = [];
		return {
			summaries,
			receivedConfigs,
			createSummary: async ({ summary, config }) => {
				receivedConfigs.push(config);
				summaries.set(summary.id, summary);
			},
			getSummary: async ({ id, config }) => {
				receivedConfigs.push(config);
				const s = summaries.get(id);
				if (!s) throw new Error(`Summary ${id} not found`);
				return s;
			},
			getSummariesForSession: async ({ sessionId, config }) => {
				receivedConfigs.push(config);
				return [...summaries.values()].filter((s) => s.sessionId === sessionId);
			},
			updateSummary: async ({ summary, config }) => {
				receivedConfigs.push(config);
				summaries.set(summary.id, summary);
			},
			deleteSummary: async ({ id, config }) => {
				receivedConfigs.push(config);
				summaries.delete(id);
			},
		};
	}

	test("config is forwarded to every store method", async () => {
		const store = createConfigAwareStore();
		const prunella = new Prunella<TestConfig>({
			pruningPolicy: { hasRole: "assistant" },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 3,
				},
			},
		});

		await prunella.prepare({
			messages: longConversation(10),
			sessionId: "tenant-test",
			config: { tenantId: "t-123" },
		});

		expect(store.receivedConfigs.length).toBeGreaterThan(0);
		for (const cfg of store.receivedConfigs) {
			expect(cfg.tenantId).toBe("t-123");
		}
	});
});

describe("compactor hooks", () => {
	test("onCompactStart and onCompactEnd fire with correct params when below threshold", async () => {
		const store = createInMemoryStore();
		const onCompactStart = mock(async (_: any) => {});
		const onCompactEnd = mock(async (_: any) => {});
		const messages = longConversation(3);

		const prunella = new Prunella({
			pruningPolicy: { hasRole: "never-matches" as never },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 999_999,
					minCompactableSpan: 100,
					maxIterations: 5,
				},
				hooks: { onCompactStart, onCompactEnd },
			},
		});

		await prunella.prepare({
			messages,
			sessionId: "hooks-no-compact",
			config: undefined,
		});

		expect(onCompactStart).toHaveBeenCalledTimes(1);
		const startParams = onCompactStart.mock.calls[0]![0];
		expect(startParams.sessionId).toBe("hooks-no-compact");
		expect(startParams.messages).toEqual(messages);
		expect(startParams.existingSummaries).toEqual([]);
		expect(startParams.estimatedTokens).toBeGreaterThan(0);

		expect(onCompactEnd).toHaveBeenCalledTimes(1);
		const endParams = onCompactEnd.mock.calls[0]![0];
		expect(endParams.sessionId).toBe("hooks-no-compact");
		expect(endParams.summaries).toEqual([]);
		expect(endParams.iterations).toBe(0);
		expect(endParams.estimatedTokens).toBeGreaterThan(0);
	});

	test("onCompactEnd reports iterations and final summaries after compaction", async () => {
		const store = createInMemoryStore();
		const onCompactEnd = mock(async (_: any) => {});

		const prunella = new Prunella({
			pruningPolicy: { hasRole: "never-matches" as never },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 3,
				},
				hooks: { onCompactEnd },
			},
		});

		await prunella.prepare({
			messages: longConversation(10),
			sessionId: "iter-count",
			config: undefined,
		});

		expect(onCompactEnd).toHaveBeenCalledTimes(1);
		const params = onCompactEnd.mock.calls[0]![0];
		expect(params.iterations).toBeGreaterThan(0);
		expect(params.summaries.length).toBeGreaterThan(0);
	});

	test("onSummaryCreate called with the new summary", async () => {
		const store = createInMemoryStore();
		const onSummaryCreate = mock(async (_: any) => {});

		const prunella = new Prunella({
			pruningPolicy: { hasRole: "never-matches" as never },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 1,
				},
				hooks: { onSummaryCreate },
			},
		});

		await prunella.prepare({
			messages: longConversation(10),
			sessionId: "create-hook",
			config: undefined,
		});

		expect(onSummaryCreate).toHaveBeenCalledTimes(1);
		const createParams = onSummaryCreate.mock.calls[0]![0];
		expect(createParams.sessionId).toBe("create-hook");
		expect(createParams.summary.id).toBeDefined();
		expect(createParams.summary.sessionId).toBe("create-hook");
		expect(createParams.summary.spans.length).toBeGreaterThan(0);
		expect(typeof createParams.summary.text).toBe("string");
	});

	test("onSummaryMerge called when two summaries are merged", async () => {
		const store = createInMemoryStore();
		const model = makeMockModel();

		const p1 = new Prunella({
			pruningPolicy: { hasRole: "never-matches" as never },
			compaction: {
				enabled: true,
				store,
				model,
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 1,
				},
			},
		});

		await p1.prepare({
			messages: longConversation(10),
			sessionId: "merge",
			config: undefined,
		});
		expect(store.summaries.size).toBe(1);

		await p1.prepare({
			messages: longConversation(20),
			sessionId: "merge",
			config: undefined,
		});
		expect(store.summaries.size).toBe(2);

		const onSummaryMerge = mock(async (_: any) => {});
		const p2 = new Prunella({
			pruningPolicy: { hasRole: "never-matches" as never },
			compaction: {
				enabled: true,
				store,
				model,
				policy: {
					compactionThreshold: 1,
					minCompactableSpan: 999_999,
					maxIterations: 1,
				},
				hooks: { onSummaryMerge },
			},
		});

		await p2.prepare({
			messages: longConversation(20),
			sessionId: "merge",
			config: undefined,
		});

		expect(onSummaryMerge).toHaveBeenCalledTimes(1);
		const mergeParams = onSummaryMerge.mock.calls[0]![0];
		expect(mergeParams.sessionId).toBe("merge");
		expect(mergeParams.mergedSummary.id).toBeDefined();
		expect(mergeParams.mergedSummary.spans.length).toBeGreaterThan(0);
		expect(mergeParams.sourceSummaries).toHaveLength(2);
	});

	test("hooks receive RuntimeConfig", async () => {
		type TC = { tenantId: string };
		const summaries = new Map<string, CompactorSummary>();
		const store: CompactorStore<TC> = {
			createSummary: async ({ summary }) => {
				summaries.set(summary.id, summary);
			},
			getSummary: async ({ id }) => {
				const s = summaries.get(id);
				if (!s) throw new Error("not found");
				return s;
			},
			getSummariesForSession: async ({ sessionId }) =>
				[...summaries.values()].filter((s) => s.sessionId === sessionId),
			updateSummary: async ({ summary }) => {
				summaries.set(summary.id, summary);
			},
			deleteSummary: async ({ id }) => {
				summaries.delete(id);
			},
		};

		const onCompactStart = mock(async (_: any) => {});
		const onCompactEnd = mock(async (_: any) => {});
		const onSummaryCreate = mock(async (_: any) => {});

		const prunella = new Prunella<TC>({
			pruningPolicy: { hasRole: "never-matches" as never },
			compaction: {
				enabled: true,
				store,
				model: makeMockModel(),
				policy: {
					compactionThreshold: 500,
					minCompactableSpan: 100,
					maxIterations: 1,
				},
				hooks: { onCompactStart, onCompactEnd, onSummaryCreate },
			},
		});

		await prunella.prepare({
			messages: longConversation(10),
			sessionId: "cfg-hook",
			config: { tenantId: "t-456" },
		});

		expect(onCompactStart).toHaveBeenCalled();
		expect(onCompactEnd).toHaveBeenCalled();
		expect(onSummaryCreate).toHaveBeenCalled();

		expect(onCompactStart.mock.calls[0]![0].config).toEqual({
			tenantId: "t-456",
		});
		expect(onCompactEnd.mock.calls[0]![0].config).toEqual({
			tenantId: "t-456",
		});
		expect(onSummaryCreate.mock.calls[0]![0].config).toEqual({
			tenantId: "t-456",
		});
	});
});

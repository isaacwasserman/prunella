import type { StandardSchemaV1 } from "@standard-schema/spec";
import {
	type LanguageModel,
	type ModelMessage,
	Output,
	generateText,
	jsonSchema,
} from "ai";
import dedent from "dedent";
import { nanoid } from "nanoid";
import { estimateTokenCount } from "tokenx";
import type { CompactorHooks } from "./hooks";
import { type PartAge, partIsOlderThan } from "./pruning";
import { renderMessages } from "./render";
import type { RuntimeConfig } from "./runtime-config";
import {
	type IdentifiableMessage,
	attachIdsToMessages,
	stripIdsFromMessages,
} from "./utils";

const SYSTEM_PROMPT =
	"You are part of a chat history compaction system. Your job is to concisely summarize a section of a message transcript, including only the most salient parts.";

export type CompactorSummary = {
	id: string;
	sessionId: string;
	spans: { firstPartId: string; lastPartId: string }[];
	text: string;
};

export interface CompactorStore<
	TRuntimeConfig extends RuntimeConfig = undefined,
> {
	createSummary: (args: {
		summary: CompactorSummary;
		config: TRuntimeConfig;
	}) => Promise<void>;
	getSummary: (args: {
		id: string;
		config: TRuntimeConfig;
	}) => Promise<CompactorSummary>;
	getSummariesForSession: (args: {
		sessionId: string;
		config: TRuntimeConfig;
	}) => Promise<CompactorSummary[]>;
	updateSummary: (args: {
		summary: CompactorSummary;
		config: TRuntimeConfig;
	}) => Promise<void>;
	deleteSummary: (args: {
		id: string;
		config: TRuntimeConfig;
	}) => Promise<void>;
}

export type CompactionOptions = {
	canCompact?: (args: {
		messages: ModelMessage[];
		messageIndex: number;
		partIndex: number;
		message: ModelMessage;
		part: ModelMessage["content"][number];
	}) => boolean;
	compactionThreshold?: number;
	minCompactableSpan?: number;
	maxIterations?: number;
	/**
	 * Parts this recent are never compacted. Defaults to `{ turns: 0 }`, which
	 * keeps the latest user message and everything after it.
	 */
	keepRecent?: PartAge;
};

export type PartSpan = { firstPartId: string; lastPartId: string };

export function summaryToMessage(summary: CompactorSummary): ModelMessage {
	return {
		role: "user",
		content: [
			{
				type: "text",
				text: dedent`
					<Summary id="${summary.id}">
						${summary.text}
					</Summary>
				`,
			},
		],
	};
}

export function getPartIndex({
	messages,
	id,
}: { messages: IdentifiableMessage[]; id: string }): {
	messageIndex: number;
	partIndex: number;
} {
	for (let mi = 0; mi < messages.length; mi++) {
		for (let pi = 0; pi < messages[mi]!.parts.length; pi++) {
			if (messages[mi]!.parts[pi]!.id === id) {
				return { messageIndex: mi, partIndex: pi };
			}
		}
	}
	throw new Error(`Part with id "${id}" not found.`);
}

export function spanIsSubspan({
	sub,
	sup,
	messages,
}: {
	sub: PartSpan;
	sup: PartSpan;
	messages: IdentifiableMessage[];
}): boolean {
	const subFirstIndex = getPartIndex({ messages, id: sub.firstPartId });
	const subLastIndex = getPartIndex({ messages, id: sub.lastPartId });
	const supFirstIndex = getPartIndex({ messages, id: sup.firstPartId });
	const supLastIndex = getPartIndex({ messages, id: sup.lastPartId });

	const startsAfter =
		subFirstIndex.messageIndex > supFirstIndex.messageIndex ||
		(subFirstIndex.messageIndex === supFirstIndex.messageIndex &&
			subFirstIndex.partIndex >= supFirstIndex.partIndex);
	const endsBefore =
		subLastIndex.messageIndex < supLastIndex.messageIndex ||
		(subLastIndex.messageIndex === supLastIndex.messageIndex &&
			subLastIndex.partIndex <= supLastIndex.partIndex);

	return startsAfter && endsBefore;
}

export function partIsCoveredBySummary({
	partId,
	summary,
	messages,
}: {
	partId: string;
	summary: CompactorSummary;
	messages: IdentifiableMessage[];
}): boolean {
	const singlePartSpan: PartSpan = { firstPartId: partId, lastPartId: partId };
	return summary.spans.some((span) =>
		spanIsSubspan({ sub: singlePartSpan, sup: span, messages }),
	);
}

export function getNextPartId({
	partId,
	messages,
}: { partId: string; messages: IdentifiableMessage[] }): string | null {
	const { messageIndex, partIndex } = getPartIndex({ messages, id: partId });
	const msg = messages[messageIndex]!;
	if (partIndex + 1 < msg.parts.length) return msg.parts[partIndex + 1]!.id;
	if (messageIndex + 1 < messages.length)
		return messages[messageIndex + 1]!.parts[0]!.id;
	return null;
}

/** Sort spans by position and join the ones that touch. */
export function mergeSpans({
	spans,
	messages,
}: { spans: PartSpan[]; messages: IdentifiableMessage[] }): PartSpan[] {
	const position = (partId: string) => {
		const { messageIndex, partIndex } = getPartIndex({ messages, id: partId });
		return [messageIndex, partIndex] as const;
	};
	const sorted = spans.toSorted((a, b) => {
		const [am, ap] = position(a.firstPartId);
		const [bm, bp] = position(b.firstPartId);
		return am !== bm ? am - bm : ap - bp;
	});
	const merged: PartSpan[] = [];
	for (const span of sorted) {
		const previous = merged.at(-1);
		if (
			previous &&
			getNextPartId({ partId: previous.lastPartId, messages }) ===
				span.firstPartId
		) {
			previous.lastPartId = span.lastPartId;
		} else {
			merged.push({ ...span });
		}
	}
	return merged;
}

export function getPartIdsInSpan({
	span,
	messages,
}: { span: PartSpan; messages: IdentifiableMessage[] }): string[] {
	const ids: string[] = [];
	let current: string | null = span.firstPartId;
	while (current) {
		ids.push(current);
		if (current === span.lastPartId) break;
		current = getNextPartId({ partId: current, messages });
	}
	return ids;
}

export class Compactor<TRuntimeConfig extends RuntimeConfig = undefined> {
	private store: CompactorStore<TRuntimeConfig>;
	private model: LanguageModel;
	private options: Required<CompactionOptions>;
	private summaryPrompt: string | undefined;
	private hooks: CompactorHooks<TRuntimeConfig> | undefined;

	constructor({
		store,
		model,
		options,
		summaryPrompt,
		hooks,
	}: {
		store: CompactorStore<TRuntimeConfig>;
		model: LanguageModel;
		options?: CompactionOptions;
		summaryPrompt?: string;
		hooks?: CompactorHooks<TRuntimeConfig>;
	}) {
		this.store = store;
		this.model = model;
		this.options = {
			canCompact: ({ message }) => {
				return message.role !== "system";
			},
			keepRecent: { turns: 0 },
			compactionThreshold: 80_000,
			minCompactableSpan: 2000,
			maxIterations: 3,
			...options,
		};
		this.summaryPrompt = summaryPrompt;
		this.hooks = hooks;
	}

	private getUncompactedSpans({
		messages,
		existingSummaries,
	}: {
		messages: IdentifiableMessage[];
		existingSummaries: CompactorSummary[];
	}): PartSpan[] {
		const spans: PartSpan[] = [];
		let currentSpan: PartSpan | null = null;
		const rawMessages = stripIdsFromMessages(messages);

		for (let mi = 0; mi < messages.length; mi++) {
			if (rawMessages[mi]!.role === "system") continue;

			for (let pi = 0; pi < messages[mi]!.parts.length; pi++) {
				const part = rawMessages[mi]!.content;
				const resolvedPart = Array.isArray(part) ? part[pi]! : part;
				const isCompactable =
					partIsOlderThan({
						messages: rawMessages,
						messageIndex: mi,
						partIndex: pi,
						ageLimit: this.options.keepRecent,
					}) &&
					this.options.canCompact({
						messages: rawMessages,
						messageIndex: mi,
						partIndex: pi,
						message: rawMessages[mi]!,
						part: resolvedPart,
					});
				const partId = messages[mi]!.parts[pi]!.id;
				const isCovered = existingSummaries.some((s) =>
					partIsCoveredBySummary({ partId, summary: s, messages }),
				);

				if (isCompactable && !isCovered) {
					if (!currentSpan) {
						currentSpan = { firstPartId: partId, lastPartId: partId };
					} else {
						currentSpan.lastPartId = partId;
					}
				} else {
					if (currentSpan) {
						spans.push(currentSpan);
						currentSpan = null;
					}
				}
			}
		}

		if (currentSpan) {
			spans.push(currentSpan);
		}

		return spans;
	}

	private getPartRange({
		span,
		messages,
	}: {
		span: PartSpan;
		messages: IdentifiableMessage[];
	}): IdentifiableMessage[] {
		const first = getPartIndex({ messages, id: span.firstPartId });
		const last = getPartIndex({ messages, id: span.lastPartId });
		return messages.slice(first.messageIndex, last.messageIndex + 1);
	}

	private sortSummaries({
		summaries,
		messages,
	}: {
		summaries: CompactorSummary[];
		messages: IdentifiableMessage[];
	}): CompactorSummary[] {
		return summaries.toSorted((a, b) => {
			const aIndex = getPartIndex({ messages, id: a.spans[0]!.firstPartId });
			const bIndex = getPartIndex({ messages, id: b.spans[0]!.firstPartId });
			if (aIndex.messageIndex !== bIndex.messageIndex)
				return aIndex.messageIndex - bIndex.messageIndex;
			return aIndex.partIndex - bIndex.partIndex;
		});
	}

	private serializeMessages(messages: IdentifiableMessage[]) {
		const serializeMessagePart = (
			part: IdentifiableMessage["parts"][number],
		) => {
			switch (part.type) {
				case "text": {
					return `<Text>${part.text}</Text>`;
				}
				case "tool-call": {
					return `<ToolCall>${JSON.stringify({ toolName: part.toolName, toolInput: part.input })}</ToolCall>`;
				}
				case "tool-result": {
					return `<ToolResult>${JSON.stringify({ toolName: part.toolName, output: part.output })}</ToolResult>`;
				}
				default: {
					return `<${part.type}>...</${part.type}>`;
				}
			}
		};

		return messages
			.map((message) => {
				return dedent`
                <${message.raw.role[0]?.toUpperCase()}${message.raw.role.slice(1)}Message>
                ${message.parts.map(serializeMessagePart).join("\n")}
                </${message.raw.role[0]?.toUpperCase()}${message.raw.role.slice(1)}Message>
            `;
			})
			.join("\n");
	}

	private createTranscriptFromSpans({
		spans,
		messages: allMessages,
	}: { spans: PartSpan[]; messages: IdentifiableMessage[] }): string {
		const messages = spans.flatMap((span) =>
			this.getPartRange({ span, messages: allMessages }),
		);

		return this.serializeMessages(messages);
	}

	private async summarizeSpans({
		spans,
		messages,
		previousSummary,
	}: {
		spans: PartSpan[];
		messages: IdentifiableMessage[];
		previousSummary?: string;
	}): Promise<string> {
		const transcript = this.createTranscriptFromSpans({ spans, messages });
		const prompt = previousSummary
			? dedent`
                <PriorSummary>
                    ${previousSummary}
                </PriorSummary>

                <Transcript>
                    ${transcript}
                </Transcript>

                The <PriorSummary> covers everything before the <Transcript>. Write one summary that replaces it. The prior summary is discarded, so anything you leave out is lost.
                - Carry forward objectives, constraints, decisions, and open work from the prior summary, even when the transcript does not mention them.
                - Where the prior summary and the transcript conflict, the transcript is newer and wins.
                - Mark work that the transcript finishes as done.
                - Simple English
                - Telegraphic style
                - Shorthand

				${this.summaryPrompt ? `Additional instructions: ${this.summaryPrompt}` : ""}
            `
			: dedent`
                <Transcript>
                    ${transcript}
                </Transcript>

                Compact the chat transcript above into a concise summary.
                - Simple English
                - Telegraphic style
                - Shorthand
                - Strictly shorter than original

				${this.summaryPrompt ? `Additional instructions: ${this.summaryPrompt}` : ""}
            `;
		const result = await generateText({
			model: this.model,
			instructions: SYSTEM_PROMPT,
			prompt,
			output: Output.object({
				schema: jsonSchema<{ summary: string }>({
					type: "object",
					properties: {
						summary: {
							type: "string",
						},
					},
					required: ["summary"],
				}),
			}),
		});
		return result.output.summary;
	}

	private async summarizeSummaries({
		summaries,
	}: {
		summaries: string[];
	}): Promise<string> {
		const result = await generateText({
			model: this.model,
			instructions: SYSTEM_PROMPT,
			prompt: dedent`
                <Summaries>
                    ${summaries.map((summary) => `<Summary>\n${summary}\n</Summary>`).join("\n")}
                </Summaries>

                Compact the sequence of chat transcript summaries above into a single concise summary.
                - Simple English
                - Telegraphic style
                - Shorthand
                - Size of a single summary

				${this.summaryPrompt ? `Additional instructions: ${this.summaryPrompt}` : ""}
            `,
			output: Output.object({
				schema: jsonSchema<{ summary: string }>({
					type: "object",
					properties: {
						summary: {
							type: "string",
						},
					},
					required: ["summary"],
				}),
			}),
		});
		return result.output.summary;
	}

	/** The session's summaries, in conversation order. */
	public async loadSummaries({
		messagesWithIds,
		sessionId,
		config,
	}: {
		messagesWithIds: IdentifiableMessage[];
		sessionId: string;
		config: TRuntimeConfig;
	}): Promise<CompactorSummary[]> {
		return this.sortSummaries({
			summaries: await this.store.getSummariesForSession({
				sessionId,
				config,
			}),
			messages: messagesWithIds,
		});
	}

	public async prepare({
		messages,
		messagesWithIds = attachIdsToMessages(messages),
		mask = new Set<string>(),
		existingSummaries: loadedSummaries,
		sessionId,
		config,
	}: {
		messages: ModelMessage[];
		/** IDs shared with the pruner. Computed from `messages` when absent. */
		messagesWithIds?: IdentifiableMessage[];
		/** Pruned part IDs. Compaction measures the conversation as rendered with them pruned. */
		mask?: Set<string>;
		/** The session's summaries, when the caller has already loaded them. */
		existingSummaries?: CompactorSummary[];
		sessionId: string;
		config: TRuntimeConfig;
	}) {
		const estimateConversationTokens = (
			existingSummaries: CompactorSummary[],
		) =>
			estimateTokenCount(
				JSON.stringify(
					renderMessages({
						messages: messagesWithIds,
						mask,
						summaries: existingSummaries,
					}),
				),
			);

		let existingSummaries =
			loadedSummaries ??
			(await this.loadSummaries({ messagesWithIds, sessionId, config }));

		let started = false;
		let summariesCreated = 0;
		let summariesMerged = 0;
		const startCompaction = async () => {
			if (started) return true;
			const proceed = await this.hooks?.onCompactStart?.({
				config,
				sessionId,
				messages,
				existingSummaries,
				estimatedTokens: estimateConversationTokens(existingSummaries),
			});
			if (proceed === false) return false;
			started = true;
			return true;
		};

		let iterations = 0;
		while (
			iterations < this.options.maxIterations &&
			estimateConversationTokens(existingSummaries) >
				this.options.compactionThreshold
		) {
			iterations++;
			const uncompactedSpans = this.getUncompactedSpans({
				messages: messagesWithIds,
				existingSummaries,
			});
			const uncompactedTokens = uncompactedSpans.reduce((total, span) => {
				const partRange = this.getPartRange({
					span,
					messages: messagesWithIds,
				});
				return (
					total +
					estimateTokenCount(
						JSON.stringify(
							renderMessages({ messages: partRange, mask, summaries: [] }),
						),
					)
				);
			}, 0);
			if (
				uncompactedSpans.length > 0 &&
				uncompactedTokens >= this.options.minCompactableSpan
			) {
				if (!(await startCompaction())) break;
				const replacedSummaries = existingSummaries;
				const summaryText = await this.summarizeSpans({
					spans: uncompactedSpans,
					messages: messagesWithIds,
					previousSummary:
						replacedSummaries.map((summary) => summary.text).join("\n\n") ||
						undefined,
				});
				const summary: CompactorSummary = {
					id: nanoid(),
					sessionId,
					spans: mergeSpans({
						spans: [
							...replacedSummaries.flatMap((summary) => summary.spans),
							...uncompactedSpans,
						],
						messages: messagesWithIds,
					}),
					text: summaryText,
				};
				await this.store.createSummary({ summary, config });
				for (const replaced of replacedSummaries) {
					await this.store.deleteSummary({ id: replaced.id, config });
				}
				summariesCreated++;
				await this.hooks?.onSummaryCreate?.({
					config,
					sessionId,
					summary,
					replacedSummaries,
				});
			} else if (existingSummaries.length > 1) {
				if (!(await startCompaction())) break;
				const sourceSummaries = existingSummaries;
				const combinedSummaryText = await this.summarizeSummaries({
					summaries: sourceSummaries.map((summary) => summary.text),
				});
				const mergedSummary: CompactorSummary = {
					id: nanoid(),
					sessionId,
					spans: mergeSpans({
						spans: sourceSummaries.flatMap((summary) => summary.spans),
						messages: messagesWithIds,
					}),
					text: combinedSummaryText,
				};
				await this.store.createSummary({ summary: mergedSummary, config });
				for (const source of sourceSummaries) {
					await this.store.deleteSummary({ id: source.id, config });
				}
				summariesMerged++;
				await this.hooks?.onSummaryMerge?.({
					config,
					sessionId,
					mergedSummary,
					sourceSummaries,
				});
			} else {
				break;
			}
			existingSummaries = this.sortSummaries({
				summaries: await this.store.getSummariesForSession({
					sessionId,
					config,
				}),
				messages: messagesWithIds,
			});
		}

		if (started) {
			await this.hooks?.onCompactEnd?.({
				config,
				sessionId,
				summaries: existingSummaries,
				estimatedTokens: estimateConversationTokens(existingSummaries),
				iterations,
				summariesCreated,
				summariesMerged,
			});
		}

		return {
			summaries: existingSummaries,
			tools: {
				// TODO: Put this tool back when it can recall part of a summary. With one
				// rolling summary, a recall returns all compacted history, which can be
				// larger than the context window.
				// "recall-summarized": tool({
				// 	description:
				// 		"Returns the original unsummarized content for a given summaryId.",
				// 	inputSchema: jsonSchema<{ summaryId: string }>({
				// 		type: "object",
				// 		properties: {
				// 			summaryId: {
				// 				type: "string",
				// 			},
				// 		},
				// 		required: ["summaryId"],
				// 	}),
				// 	execute: (input) => {
				// 		const summary = existingSummaries.find(
				// 			(summary) => summary.id === input.summaryId,
				// 		);
				// 		if (!summary) {
				// 			throw new Error(`No summary found with id "${input.summaryId}"`);
				// 		}
				// 		const transcript = this.createTranscriptFromSpans({
				// 			spans: summary.spans,
				// 			messages: messagesWithIds,
				// 		});
				// 		return transcript;
				// 	},
				// }),
			},
		};
	}
}

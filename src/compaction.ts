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
import {
	type IdentifiableMessage,
	attachIdsToMessages,
	stripIdsFromMessages,
} from "./utils";

const SYSTEM_PROMPT =
	"You are part of a chat history compaction system. Your job is to concisely summarize a section of a message transcript, including only the most salient parts.";
const COMBINED_SUMMARY_SENTINEL =
	"This summary has been combined with subsequent summaries.";

export type CompactorSummary = {
	id: string;
	sessionId: string;
	firstPartId: string;
	lastPartId: string;
	collapsed: boolean;
	text: string;
};

export interface CompactorStore {
	createSummary: (args: { summary: CompactorSummary }) => Promise<void>;
	getSummary: (args: { id: string }) => Promise<CompactorSummary>;
	getSummariesForSession: (args: { sessionId: string }) => Promise<
		CompactorSummary[]
	>;
	updateSummary: (args: { summary: CompactorSummary }) => Promise<void>;
	deleteSummary: (args: { id: string }) => Promise<void>;
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
};

export type PartSpan = { firstPartId: string; lastPartId: string };

export function summaryToMessage(summary: CompactorSummary): ModelMessage {
	return {
		role: "user",
		content: [
			{
				type: "text",
				text: dedent`
                <Summary>
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

export class Compactor {
	private store: CompactorStore;
	private model: LanguageModel;
	private policy: Required<CompactionOptions>;

	constructor({
		store,
		model,
		policy,
	}: {
		store: CompactorStore;
		model: LanguageModel;
		policy: CompactionOptions;
	}) {
		this.store = store;
		this.model = model;
		this.policy = {
			canCompact: ({ message }) => {
				return message.role !== "system";
			},
			compactionThreshold: 80_000,
			minCompactableSpan: 2000,
			maxIterations: 3,
			...policy,
		};
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
				const isCompactable = this.policy.canCompact({
					messages: rawMessages,
					messageIndex: mi,
					partIndex: pi,
					message: rawMessages[mi]!,
					part: resolvedPart,
				});
				const partId = messages[mi]!.parts[pi]!.id;
				const singlePartSpan: PartSpan = {
					firstPartId: partId,
					lastPartId: partId,
				};
				const isCovered = existingSummaries.some((s) =>
					spanIsSubspan({ sub: singlePartSpan, sup: s, messages }),
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

	private findSummaryAt({
		partId,
		span,
		messages,
		summaries,
	}: {
		partId: string;
		span: PartSpan;
		messages: IdentifiableMessage[];
		summaries: CompactorSummary[];
	}): CompactorSummary | undefined {
		return summaries.find(
			(s) =>
				s.firstPartId === partId &&
				spanIsSubspan({ sub: s, sup: span, messages }),
		);
	}

	private fullSpan(messages: IdentifiableMessage[]): PartSpan {
		const firstPartId = messages[0]!.parts[0]!.id;
		const lastMsg = messages[messages.length - 1]!;
		const lastPartId = lastMsg.parts[lastMsg.parts.length - 1]!.id;
		return { firstPartId, lastPartId };
	}

	private interpolateSummaries({
		span,
		messages,
		existingSummaries,
	}: {
		span?: PartSpan;
		messages: IdentifiableMessage[];
		existingSummaries: CompactorSummary[];
	}): ModelMessage[] {
		const resolvedSpan = span ?? this.fullSpan(messages);
		const partIds = getPartIdsInSpan({ span: resolvedSpan, messages });
		const result: ModelMessage[] = [];
		let rawStartIndex: number | null = null;
		let i = 0;

		const flushRawParts = () => {
			if (rawStartIndex !== null) {
				const rawSpan: PartSpan = {
					firstPartId: partIds[rawStartIndex]!,
					lastPartId: partIds[i - 1]!,
				};
				result.push(
					...stripIdsFromMessages(
						this.getPartRange({ span: rawSpan, messages }),
					),
				);
				rawStartIndex = null;
			}
		};

		while (i < partIds.length) {
			const summary = this.findSummaryAt({
				partId: partIds[i]!,
				span: resolvedSpan,
				messages,
				summaries: existingSummaries,
			});
			if (summary) {
				flushRawParts();
				result.push(summaryToMessage(summary));
				i = partIds.indexOf(summary.lastPartId) + 1;
			} else {
				if (rawStartIndex === null) rawStartIndex = i;
				i++;
			}
		}
		flushRawParts();

		return result;
	}

	private sortSummaries({
		summaries,
		messages,
	}: {
		summaries: CompactorSummary[];
		messages: IdentifiableMessage[];
	}): CompactorSummary[] {
		return summaries.toSorted((a, b) => {
			const aIndex = getPartIndex({ messages, id: a.firstPartId });
			const bIndex = getPartIndex({ messages, id: b.firstPartId });
			if (aIndex.messageIndex !== bIndex.messageIndex)
				return aIndex.messageIndex - bIndex.messageIndex;
			return aIndex.partIndex - bIndex.partIndex;
		});
	}

	private createTranscript({
		spans,
		messages: allMessages,
	}: { spans: PartSpan[]; messages: IdentifiableMessage[] }): string {
		const messages = spans.flatMap((span) =>
			this.getPartRange({ span, messages: allMessages }),
		);

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
					return `<ToolResult>${JSON.stringify({ toolName: part.toolName, toolInput: part.output })}</ToolResult>`;
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

	private async summarizeSpans({
		spans,
		messages,
	}: {
		spans: PartSpan[];
		messages: IdentifiableMessage[];
	}): Promise<string> {
		const result = await generateText({
			model: this.model,
			instructions: SYSTEM_PROMPT,
			prompt: dedent`
                <Transcript>
                    ${this.createTranscript({ spans, messages })}
                </Transcript>

                Compact the chat transcript above into a concise summary.
                - Simple English
                - Telegraphic style
                - Shorthand
                - Strictly shorter than original
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

	public async prepare({
		messages,
		sessionId,
	}: { messages: ModelMessage[]; sessionId: string }) {
		const messagesWithIds = attachIdsToMessages(messages);

		const estimateConversationTokens = async (
			existingSummaries: CompactorSummary[],
		) => {
			const messagesWithSummaries = this.interpolateSummaries({
				messages: messagesWithIds,
				existingSummaries,
			});
			const tokenCount = estimateTokenCount(
				JSON.stringify(messagesWithSummaries),
			);
			return tokenCount;
		};

		let existingSummaries = this.sortSummaries({
			summaries: await this.store.getSummariesForSession({
				sessionId,
			}),
			messages: messagesWithIds,
		});

		let iterations = 0;
		while (
			iterations < this.policy.maxIterations &&
			(await estimateConversationTokens(existingSummaries)) >
				this.policy.compactionThreshold
		) {
			iterations++;
			const [combinableSummary1, combinableSummary2] = existingSummaries.filter(
				(summary) => !summary.collapsed,
			);
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
					estimateTokenCount(JSON.stringify(stripIdsFromMessages(partRange)))
				);
			}, 0);
			if (
				uncompactedSpans.length > 0 &&
				uncompactedTokens >= this.policy.minCompactableSpan
			) {
				// Compact spans
				const summaryText = await this.summarizeSpans({
					spans: uncompactedSpans,
					messages: messagesWithIds,
				});
				let lastGroupStart = uncompactedSpans.length - 1;
				for (let j = uncompactedSpans.length - 1; j > 0; j--) {
					const nextPartAfterPrev = getNextPartId({
						partId: uncompactedSpans[j - 1]!.lastPartId,
						messages: messagesWithIds,
					});
					if (nextPartAfterPrev === uncompactedSpans[j]!.firstPartId) {
						lastGroupStart = j - 1;
					} else {
						break;
					}
				}
				await this.store.createSummary({
					summary: {
						id: nanoid(),
						sessionId,
						firstPartId: uncompactedSpans[lastGroupStart]!.firstPartId,
						lastPartId:
							uncompactedSpans[uncompactedSpans.length - 1]!.lastPartId,
						text: summaryText,
						collapsed: false,
					},
				});
			} else if (combinableSummary1 && combinableSummary2) {
				// Combine first and second summaries
				const combinedSummaryText = await this.summarizeSummaries({
					summaries: [combinableSummary1.text, combinableSummary2.text],
				});
				const consecutive =
					getNextPartId({
						partId: combinableSummary1.lastPartId,
						messages: messagesWithIds,
					}) === combinableSummary2.firstPartId;
				if (consecutive) {
					await this.store.createSummary({
						summary: {
							id: nanoid(),
							sessionId,
							firstPartId: combinableSummary1.firstPartId,
							lastPartId: combinableSummary2.lastPartId,
							text: combinedSummaryText,
							collapsed: false,
						},
					});
					await this.store.deleteSummary({
						id: combinableSummary1.id,
					});
					await this.store.deleteSummary({
						id: combinableSummary2.id,
					});
				} else {
					await this.store.updateSummary({
						summary: {
							...combinableSummary1,
							text: COMBINED_SUMMARY_SENTINEL,
							collapsed: true,
						},
					});
					await this.store.updateSummary({
						summary: {
							...combinableSummary2,
							text: combinedSummaryText,
							collapsed: false,
						},
					});
				}
			} else {
				break;
			}
			existingSummaries = this.sortSummaries({
				summaries: await this.store.getSummariesForSession({
					sessionId,
				}),
				messages: messagesWithIds,
			});
		}

		return { summaries: existingSummaries };
	}
}

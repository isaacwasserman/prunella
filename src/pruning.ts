import { type ModelMessage, type ToolCallPart, jsonSchema, tool } from "ai";
import {
	type IdentifiableMessage,
	type TokensAfter,
	getMessageByIndex,
	getPartByIndex,
	partTokens,
	stripIdsFromMessages,
	tokensAfterParts,
} from "./utils";

const RECALL_TOOL_NAME = "recall-pruned";

export type PartAge =
	| {
			turns: number;
	  }
	| {
			steps: number;
	  }
	| {
			messages: number;
	  }
	| {
			/**
			 * Tokens of the parts after it. A pruning policy counts their original
			 * content; compaction's `keepRecent` counts them as rendered, pruned.
			 */
			tokens: number;
	  };

export const DEFAULT_PRESSURE_BUFFER_FACTOR = 0.1;

/**
 * True for the parts chosen to relieve pressure. When the rendered
 * conversation is over `budget` tokens, the oldest parts that the rest of the
 * policy would prune with this condition true are chosen, in steps of
 * `bufferFactor * budget` tokens, so the conversation drops to about
 * `budget * (1 - bufferFactor)` and grows back to `budget` before more is
 * chosen.
 */
export type PressureCondition = {
	budget: number;
	bufferFactor?: number;
};

/** How the pruner measures the conversation it renders. */
export type PressureMeasure = {
	/** Tokens of the rendered conversation with `mask` pruned. */
	size: (mask: Set<string>) => number;
	/** Tokens saved by pruning one part. */
	savings: (partId: string) => number;
};

type PrunePredicate = ({
	messageIndex,
	partIndex,
	messages,
	message,
	part,
}: {
	messageIndex: number;
	partIndex: number;
	messages: ModelMessage[];
	message: ModelMessage;
	part: string | ModelMessage["content"][number];
}) => boolean;

export type PruningPolicy =
	| {
			OR: PruningPolicy[];
	  }
	| {
			AND: PruningPolicy[];
	  }
	| {
			NOT: PruningPolicy;
	  }
	| {
			olderThan: PartAge;
	  }
	| {
			hasType: Exclude<ModelMessage["content"][number], string>["type"];
	  }
	| {
			hasRole: ModelMessage["role"];
	  }
	| {
			shouldPrune: PrunePredicate;
	  }
	| {
			hasPressure: PressureCondition;
	  };

type AllPruningPolicyKeys = keyof {
	[K in PruningPolicy as keyof K]: true;
};

function serializePart(part: string | ModelMessage["content"][number]): string {
	if (typeof part === "string") return part;
	switch (part.type) {
		case "text":
			return part.text;
		case "tool-call":
			return JSON.stringify({
				toolName: (part as ToolCallPart).toolName,
				input: (part as ToolCallPart).input,
			});
		case "tool-result":
			return JSON.stringify({
				toolName: (part as { toolName: string }).toolName,
				output: (part as { output: unknown }).output,
			});
		default:
			return JSON.stringify(part);
	}
}

/** Whether a part is older than `ageLimit`, counted in user turns, parts, messages, or tokens after it. */
export function partIsOlderThan({
	messages,
	messageIndex,
	partIndex,
	ageLimit,
	tokensAfter,
}: {
	messages: ModelMessage[];
	messageIndex: number;
	partIndex: number;
	ageLimit: PartAge;
	/** Needed for `{ tokens }`. */
	tokensAfter?: TokensAfter;
}): boolean {
	if ("turns" in ageLimit) {
		let turnAge = 0;
		const mostRecentMessageIndex = messages.length - 1;
		if (messageIndex < mostRecentMessageIndex) {
			const mostRecentMessage = getMessageByIndex({
				messages,
				messageIndex: mostRecentMessageIndex,
			});
			let subsequentMessageRole: ModelMessage["role"] = mostRecentMessage.role;
			for (
				let messageIndexCursor = mostRecentMessageIndex - 1;
				messageIndexCursor >= messageIndex;
				messageIndexCursor--
			) {
				const messageRole = getMessageByIndex({
					messages,
					messageIndex: messageIndexCursor,
				}).role;
				if (subsequentMessageRole === "user" && messageRole !== "user") {
					turnAge++;
				}
				subsequentMessageRole = messageRole;
			}
		}

		return turnAge > ageLimit.turns;
	}
	if ("steps" in ageLimit) {
		let partAge = 0;
		for (
			let messageIndexCursor = messageIndex;
			messageIndexCursor < messages.length;
			messageIndexCursor++
		) {
			const messageContent = getMessageByIndex({
				messages,
				messageIndex: messageIndexCursor,
			}).content;
			const messageContentLength = Array.isArray(messageContent)
				? messageContent.length
				: 1;
			if (messageIndexCursor === messageIndex) {
				partAge += messageContentLength - (partIndex + 1);
			} else {
				partAge += messageContentLength;
			}
		}

		return partAge > ageLimit.steps;
	}
	if ("messages" in ageLimit) {
		return messages.length - 1 - messageIndex > ageLimit.messages;
	}
	if ("tokens" in ageLimit) {
		if (!tokensAfter) throw new Error("A tokens part age needs part sizes");
		return tokensAfter(messageIndex, partIndex) > ageLimit.tokens;
	}
	throw new Error("Invalid part age construction");
}

function collectPressureConditions(policy: PruningPolicy): PressureCondition[] {
	if ("hasPressure" in policy) {
		const { budget, bufferFactor = DEFAULT_PRESSURE_BUFFER_FACTOR } =
			policy.hasPressure;
		if (!Number.isFinite(budget) || budget < 0) {
			throw new Error("hasPressure budget must be a finite number >= 0");
		}
		if (!(bufferFactor >= 0 && bufferFactor < 1)) {
			throw new Error("hasPressure bufferFactor must be >= 0 and < 1");
		}
		return [policy.hasPressure];
	}
	if ("AND" in policy) return policy.AND.flatMap(collectPressureConditions);
	if ("OR" in policy) return policy.OR.flatMap(collectPressureConditions);
	if ("NOT" in policy) return collectPressureConditions(policy.NOT);
	return [];
}

export class Pruner {
	private pruningPolicy: PruningPolicy;
	private pressureConditions: PressureCondition[];

	constructor(args: { pruningPolicy: PruningPolicy }) {
		this.pruningPolicy = args.pruningPolicy;
		this.pressureConditions = collectPressureConditions(args.pruningPolicy);
	}

	private evaluatePruningPolicy({
		messages,
		policyFragment,
		messageIndex,
		partIndex,
		partId,
		chosen,
		tokensAfter,
	}: {
		messages: ModelMessage[];
		policyFragment: PruningPolicy;
		messageIndex: number;
		partIndex: number;
		partId: string;
		chosen: Map<PressureCondition, Set<string>>;
		tokensAfter: TokensAfter;
	}): boolean {
		const policyKeys = Object.keys(policyFragment) as AllPruningPolicyKeys[];
		return policyKeys.every((policyKey) => {
			switch (policyKey) {
				case "AND": {
					const subPolicies = policyFragment[
						policyKey as keyof typeof policyFragment
					] as PruningPolicy[];
					return subPolicies.every((subPolicy) =>
						this.evaluatePruningPolicy({
							messages,
							policyFragment: subPolicy,
							messageIndex,
							partIndex,
							partId,
							chosen,
							tokensAfter,
						}),
					);
				}
				case "OR": {
					const subPolicies = policyFragment[
						policyKey as keyof typeof policyFragment
					] as PruningPolicy[];
					return subPolicies.some((subPolicy) =>
						this.evaluatePruningPolicy({
							messages,
							policyFragment: subPolicy,
							messageIndex,
							partIndex,
							partId,
							chosen,
							tokensAfter,
						}),
					);
				}
				case "NOT": {
					const subPolicy = policyFragment[
						policyKey as keyof typeof policyFragment
					] as PruningPolicy;
					return !this.evaluatePruningPolicy({
						messages,
						policyFragment: subPolicy,
						messageIndex,
						partIndex,
						partId,
						chosen,
						tokensAfter,
					});
				}
				case "olderThan": {
					const ageLimit = (policyFragment as { olderThan: PartAge }).olderThan;
					return partIsOlderThan({
						messages,
						messageIndex,
						partIndex,
						ageLimit,
						tokensAfter,
					});
				}
				case "hasRole": {
					const message = getMessageByIndex({ messages, messageIndex });
					const targetRole = (
						policyFragment as { hasRole: ModelMessage["role"] }
					).hasRole;
					return message.role === targetRole;
				}
				case "hasType": {
					const part = getPartByIndex({
						messages,
						messageIndex,
						partIndex,
					});
					const targetType = (
						policyFragment as {
							hasType: Exclude<ModelMessage["content"][number], string>["type"];
						}
					).hasType;
					const partType = typeof part === "string" ? "text" : part.type;
					return partType === targetType;
				}
				case "shouldPrune": {
					const predicate = (policyFragment as { shouldPrune: PrunePredicate })
						.shouldPrune;
					const message = getMessageByIndex({ messages, messageIndex });
					const part = getPartByIndex({
						messages,
						messageIndex,
						partIndex,
					});
					return predicate({
						messageIndex,
						partIndex,
						messages,
						message,
						part,
					});
				}
				case "hasPressure": {
					const condition = (
						policyFragment as { hasPressure: PressureCondition }
					).hasPressure;
					return chosen.get(condition)?.has(partId) ?? false;
				}
			}
		});
	}

	private partIsRecallRequest({
		messages,
		messageIndex,
		partIndex,
	}: {
		messages: ModelMessage[];
		messageIndex: number;
		partIndex: number;
	}): boolean {
		const part = getPartByIndex({ messages, messageIndex, partIndex });
		if (
			typeof part !== "string" &&
			part.type === "tool-call" &&
			part.toolName === RECALL_TOOL_NAME
		) {
			return true;
		}
		return false;
	}

	private partIsPruned({
		messages,
		messageIndex,
		partIndex,
		partId,
		chosen,
		tokensAfter,
	}: {
		messages: ModelMessage[];
		messageIndex: number;
		partIndex: number;
		partId: string;
		chosen: Map<PressureCondition, Set<string>>;
		tokensAfter: TokensAfter;
	}): boolean {
		return (
			this.evaluatePruningPolicy({
				messages,
				policyFragment: this.pruningPolicy,
				messageIndex,
				partIndex,
				partId,
				chosen,
				tokensAfter,
			}) && !this.partIsRecallRequest({ messages, messageIndex, partIndex })
		);
	}

	/**
	 * Choose parts for each `hasPressure` condition in policy order. A part is a
	 * candidate when choosing it makes the policy prune it; candidates are taken
	 * oldest first until they save enough.
	 */
	private choosePressureParts({
		messages,
		parts,
		measure,
		tokensAfter,
	}: {
		messages: ModelMessage[];
		parts: { messageIndex: number; partIndex: number; partId: string }[];
		measure: PressureMeasure | undefined;
		tokensAfter: TokensAfter;
	}): Map<PressureCondition, Set<string>> {
		const chosen = new Map<PressureCondition, Set<string>>(
			this.pressureConditions.map((condition) => [condition, new Set()]),
		);
		const isPruned = (part: (typeof parts)[number]) =>
			this.partIsPruned({ messages, ...part, chosen, tokensAfter });
		const currentMask = () =>
			new Set(parts.filter(isPruned).map((part) => part.partId));

		if (!measure) return chosen;
		for (const condition of this.pressureConditions) {
			const mask = currentMask();
			const over = measure.size(mask) - condition.budget;
			if (over <= 0) continue;
			const band =
				(condition.bufferFactor ?? DEFAULT_PRESSURE_BUFFER_FACTOR) *
				condition.budget;
			const required = band > 0 ? Math.ceil(over / band) * band : over;
			const conditionParts = chosen.get(condition)!;
			let saved = 0;
			for (const part of parts) {
				if (saved >= required) break;
				if (mask.has(part.partId)) continue;
				const savings = measure.savings(part.partId);
				if (savings <= 0) continue;
				conditionParts.add(part.partId);
				if (isPruned(part)) {
					saved += savings;
				} else {
					conditionParts.delete(part.partId);
				}
			}
		}
		return chosen;
	}

	public prepare({
		messages: identifiableMessages,
		measure,
	}: {
		messages: IdentifiableMessage[];
		/** Needed for `hasPressure`; without it, no part is under pressure. */
		measure?: PressureMeasure;
	}) {
		const messages = stripIdsFromMessages(identifiableMessages);
		const mask = new Set<string>();
		const originalContent = new Map<string, string>();
		const parts = identifiableMessages.flatMap((message, messageIndex) =>
			message.parts.map((part, partIndex) => ({
				messageIndex,
				partIndex,
				partId: part.id,
			})),
		);
		let sizes: TokensAfter | undefined;
		const tokensAfter: TokensAfter = (messageIndex, partIndex) => {
			sizes ??= tokensAfterParts(identifiableMessages, partTokens);
			return sizes(messageIndex, partIndex);
		};
		const chosen = this.choosePressureParts({
			messages,
			parts,
			measure,
			tokensAfter,
		});

		for (const part of parts) {
			if (this.partIsPruned({ messages, ...part, chosen, tokensAfter })) {
				mask.add(part.partId);
				originalContent.set(
					part.partId,
					serializePart(
						getPartByIndex({
							messages,
							messageIndex: part.messageIndex,
							partIndex: part.partIndex,
						}),
					),
				);
			}
		}

		const recallTool = tool({
			description: "Recall pruned part.",
			inputSchema: jsonSchema({
				type: "object",
				properties: {
					pruneId: {
						type: "string",
						description: "The pruneId of the part to recall.",
					},
				},
				required: ["pruneId"],
			}),
			execute: (input) => {
				const content = originalContent.get(
					(input as { pruneId: string }).pruneId,
				);
				if (!content) return "Part not found or not pruned.";
				return content;
			},
		});

		return {
			mask,
			tools: { [RECALL_TOOL_NAME]: recallTool },
		};
	}
}

import { type ModelMessage, type ToolCallPart, jsonSchema, tool } from "ai";
import type { RuntimeConfig } from "./runtime-config";
import {
	type IdentifiableMessage,
	getMessageByIndex,
	getPartByIndex,
	stripIdsFromMessages,
} from "./utils";

const RECALL_TOOL_NAME = "recall-pruned";

export type PartAge =
	| {
			turns: number;
	  }
	| {
			steps: number;
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

export class Pruner {
	private pruningPolicy: PruningPolicy;

	constructor(args: { pruningPolicy: PruningPolicy }) {
		this.pruningPolicy = args.pruningPolicy;
	}

	private partOlderThan({
		messages,
		messageIndex,
		partIndex,
		ageLimit,
	}: {
		messages: ModelMessage[];
		messageIndex: number;
		partIndex: number;
		ageLimit: PartAge;
	}): boolean {
		if ("turns" in ageLimit) {
			let turnAge = 0;
			const mostRecentMessageIndex = messages.length - 1;
			if (messageIndex < mostRecentMessageIndex) {
				const mostRecentMessage = getMessageByIndex({
					messages,
					messageIndex: mostRecentMessageIndex,
				});
				let subsequentMessageRole: ModelMessage["role"] =
					mostRecentMessage.role;
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
		throw new Error("Invalid part age construction");
	}

	private evaluatePruningPolicy({
		messages,
		policyFragment,
		messageIndex,
		partIndex,
	}: {
		messages: ModelMessage[];
		policyFragment: PruningPolicy;
		messageIndex: number;
		partIndex: number;
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
					});
				}
				case "olderThan": {
					const ageLimit = (policyFragment as { olderThan: PartAge }).olderThan;
					return this.partOlderThan({
						messages,
						messageIndex,
						partIndex,
						ageLimit,
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
	}: {
		messages: ModelMessage[];
		messageIndex: number;
		partIndex: number;
	}): boolean {
		return (
			this.evaluatePruningPolicy({
				messages,
				policyFragment: this.pruningPolicy,
				messageIndex,
				partIndex,
			}) && !this.partIsRecallRequest({ messages, messageIndex, partIndex })
		);
	}

	public prepare({
		messages: identifiableMessages,
	}: { messages: IdentifiableMessage[] }) {
		const messages = stripIdsFromMessages(identifiableMessages);
		const mask = new Set<string>();
		const originalContent = new Map<string, string>();

		for (let mi = 0; mi < identifiableMessages.length; mi++) {
			for (let pi = 0; pi < identifiableMessages[mi]!.parts.length; pi++) {
				if (this.partIsPruned({ messages, messageIndex: mi, partIndex: pi })) {
					const partId = identifiableMessages[mi]!.parts[pi]!.id;
					mask.add(partId);
					const part = getPartByIndex({
						messages,
						messageIndex: mi,
						partIndex: pi,
					});
					originalContent.set(partId, serializePart(part));
				}
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

import { createHash } from "node:crypto";
import {
	type ModelMessage,
	type TextPart,
	type ToolCallPart,
	jsonSchema,
	tool,
} from "ai";
import dedent from "dedent";

const PRUNE_TOOL_RESULTS_OLDER_THAN_N_TURNS = 2;

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
		return "Recalled.";
	},
});

export type RecallPolicy = {
	recallDuration: PartAge;
};

export class Prunella {
	private messages: ModelMessage[];
	private pruningPolicy: PruningPolicy;
	private recallPolicy: RecallPolicy;

	constructor(
		messages: ModelMessage[],
		pruningPolicy: PruningPolicy,
		recallPolicy: RecallPolicy,
	) {
		this.messages = messages;
		this.pruningPolicy = pruningPolicy;
		this.recallPolicy = recallPolicy;
	}

	private static hashString(str: string) {
		return createHash("sha256").update(str).digest("hex");
	}

	private getMessagePartIdentity({
		messageIndex,
		partIndex,
	}: { messageIndex: number; partIndex: number }): string {
		const previousMessage =
			messageIndex > 0
				? this.getMessageByIndex({ messageIndex: messageIndex - 1 })
				: null;
		const previousMessagePart =
			partIndex > 0
				? this.getPartByIndex({ messageIndex, partIndex: partIndex - 1 })
				: null;
		const targetPart = this.getPartByIndex({ messageIndex, partIndex });

		const identityHash = Prunella.hashString(
			JSON.stringify(previousMessage) +
				JSON.stringify(previousMessagePart) +
				JSON.stringify(targetPart),
		);

		return identityHash;
	}

	private getMessageByIndex({
		messageIndex,
	}: { messageIndex: number }): ModelMessage {
		const message = this.messages[messageIndex];
		if (!message) {
			throw new Error(
				`Looked for message at index ${messageIndex} but none was found.`,
			);
		}
		return message;
	}

	private getPartByIndex({
		messageIndex,
		partIndex,
	}: { messageIndex: number; partIndex: number }) {
		const message = this.getMessageByIndex({ messageIndex });
		const content = message.content;
		if (!Array.isArray(content)) {
			if (partIndex > 0) {
				throw new Error(
					`Looked for message part at index ${messageIndex} but message content is not an array.`,
				);
			}
			return content;
		}
		const part = content[partIndex];
		if (!part) {
			throw new Error(
				`Looked for message part at index ${partIndex} but none was found.`,
			);
		}
		return part;
	}

	private createMessagePartPlaceholder({
		messageIndex,
		partIndex,
	}: { messageIndex: number; partIndex: number }): string | TextPart {
		const part = this.getPartByIndex({ messageIndex, partIndex });

		const disclaimer = dedent`
            This part of the message has been pruned for token efficiency. To have its content revealed, run the ${RECALL_TOOL_NAME} tool with pruneId "${this.getMessagePartIdentity({ messageIndex, partIndex })}".
        `;

		if (typeof part === "string") {
			return disclaimer;
		}

		return {
			type: "text",
			text: disclaimer,
		};
	}

	private partOlderThan({
		messageIndex,
		partIndex,
		ageLimit,
	}: { messageIndex: number; partIndex: number; ageLimit: PartAge }): boolean {
		if ("turns" in ageLimit) {
			let turnAge = 0;
			const mostRecentMessageIndex = this.messages.length - 1;
			if (messageIndex < mostRecentMessageIndex) {
				const mostRecentMessage = this.getMessageByIndex({
					messageIndex: mostRecentMessageIndex,
				});
				let subsequentMessageRole: ModelMessage["role"] =
					mostRecentMessage.role;
				for (
					let messageIndexCursor = mostRecentMessageIndex - 1;
					messageIndexCursor >= messageIndex;
					messageIndexCursor--
				) {
					const messageRole = this.getMessageByIndex({
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
				messageIndexCursor < this.messages.length;
				messageIndexCursor++
			) {
				const messageContent = this.getMessageByIndex({
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
		policyFragment,
		messageIndex,
		partIndex,
	}: {
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
						policyFragment: subPolicy,
						messageIndex,
						partIndex,
					});
				}
				case "olderThan": {
					const ageLimit = (policyFragment as { olderThan: PartAge }).olderThan;
					return this.partOlderThan({ messageIndex, partIndex, ageLimit });
				}
				case "hasRole": {
					const message = this.getMessageByIndex({ messageIndex });
					const targetRole = (
						policyFragment as { hasRole: ModelMessage["role"] }
					).hasRole;
					return message.role === targetRole;
				}
				case "hasType": {
					const part = this.getPartByIndex({ messageIndex, partIndex });
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
					const message = this.getMessageByIndex({ messageIndex });
					const part = this.getPartByIndex({ messageIndex, partIndex });
					return predicate({
						messageIndex,
						partIndex,
						messages: this.messages,
						message,
						part,
					});
				}
			}
		});
	}

	private partIsRecallRequest({
		messageIndex,
		partIndex,
	}: { messageIndex: number; partIndex: number }): boolean {
		const part = this.getPartByIndex({ messageIndex, partIndex });
		if (
			typeof part !== "string" &&
			part.type === "tool-call" &&
			part.toolName === RECALL_TOOL_NAME
		) {
			return true;
		}
		return false;
	}

	private getRecallTargetId({ toolPart }: { toolPart: ToolCallPart }): string {
		const pruneId = (toolPart.input as { pruneId: string }).pruneId;
		return pruneId;
	}

	private partIsRecalled({
		messageIndex,
		partIndex,
	}: { messageIndex: number; partIndex: number }): boolean {
		const pruneId = this.getMessagePartIdentity({ messageIndex, partIndex });
		for (
			let messageIndexCursor = this.messages.length - 1;
			messageIndexCursor >= messageIndex;
			messageIndexCursor--
		) {
			const message = this.getMessageByIndex({
				messageIndex: messageIndexCursor,
			});
			for (
				let partIndexCursor =
					typeof message.content === "string" ? 0 : message.content.length - 1;
				partIndexCursor >= 0;
				partIndexCursor--
			) {
				if (
					this.partIsRecallRequest({
						messageIndex: messageIndexCursor,
						partIndex: partIndexCursor,
					})
				) {
					const recallRequestPart = this.getPartByIndex({
						messageIndex: messageIndexCursor,
						partIndex: partIndexCursor,
					}) as ToolCallPart;
					const recalledId = this.getRecallTargetId({
						toolPart: recallRequestPart,
					});
					if (recalledId === pruneId) {
						if (
							!this.partOlderThan({
								messageIndex: messageIndexCursor,
								partIndex: partIndexCursor,
								ageLimit: this.recallPolicy.recallDuration,
							})
						) {
							return true;
						}
						return false;
					}
				}
			}
		}
		return false;
	}

	private partIsPruned({
		messageIndex,
		partIndex,
	}: { messageIndex: number; partIndex: number }): boolean {
		return (
			this.evaluatePruningPolicy({
				policyFragment: this.pruningPolicy,
				messageIndex,
				partIndex,
			}) &&
			!this.partIsRecalled({ messageIndex, partIndex }) &&
			!this.partIsRecallRequest({ messageIndex, partIndex })
		);
	}

	public prepare() {
		const prunedMessages = this.messages.map((message, messageIndex) => {
			const content = message.content;
			if (!Array.isArray(content)) {
				return {
					...message,
					content: this.partIsPruned({ messageIndex, partIndex: 0 })
						? (this.createMessagePartPlaceholder({
								messageIndex,
								partIndex: 0,
							}) as string)
						: content,
				} as ModelMessage;
			}
			return {
				...message,
				content: content.map((part, partIndex) =>
					this.partIsPruned({ messageIndex, partIndex })
						? this.createMessagePartPlaceholder({ messageIndex, partIndex })
						: part,
				),
			} as ModelMessage;
		});

		return {
			messages: prunedMessages,
			tools: { [RECALL_TOOL_NAME]: recallTool },
		};
	}
}

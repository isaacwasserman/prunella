import { createHash } from "node:crypto";
import type { ModelMessage } from "ai";

export type IdentifiablePart = Exclude<
	ModelMessage["content"],
	string
>[number] & {
	id: string;
};

export type IdentifiableMessage = {
	id: string;
	raw: ModelMessage;
	parts: IdentifiablePart[];
};

export function hashString(str: string): string {
	return createHash("sha256").update(str).digest("hex");
}

export function getMessageByIndex({
	messages,
	messageIndex,
}: { messages: ModelMessage[]; messageIndex: number }): ModelMessage {
	const message = messages[messageIndex];
	if (!message) {
		throw new Error(
			`Looked for message at index ${messageIndex} but none was found.`,
		);
	}
	return message;
}

export function getPartByIndex({
	messages,
	messageIndex,
	partIndex,
}: { messages: ModelMessage[]; messageIndex: number; partIndex: number }) {
	const message = getMessageByIndex({ messages, messageIndex });
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

export function getMessageIdentity({
	messages,
	messageIndex,
}: { messages: ModelMessage[]; messageIndex: number }): string {
	const previousMessage =
		messageIndex > 0
			? getMessageByIndex({ messages, messageIndex: messageIndex - 1 })
			: null;
	const targetMessage = getMessageByIndex({
		messages,
		messageIndex,
	});
	const identityHash = hashString(
		JSON.stringify(previousMessage) + JSON.stringify(targetMessage),
	);
	return identityHash;
}

export function getMessagePartIdentity({
	messages,
	messageIndex,
	partIndex,
}: {
	messages: ModelMessage[];
	messageIndex: number;
	partIndex: number;
}): string {
	const previousMessage =
		messageIndex > 0
			? getMessageByIndex({ messages, messageIndex: messageIndex - 1 })
			: null;
	const previousMessagePart =
		partIndex > 0
			? getPartByIndex({
					messages,
					messageIndex,
					partIndex: partIndex - 1,
				})
			: null;
	const targetPart = getPartByIndex({
		messages,
		messageIndex,
		partIndex,
	});

	const identityHash = hashString(
		JSON.stringify(previousMessage) +
			JSON.stringify(previousMessagePart) +
			JSON.stringify(targetPart),
	);

	return identityHash;
}

export function attachIdsToParts({
	messages,
	messageIndex,
}: { messages: ModelMessage[]; messageIndex: number }): IdentifiablePart[] {
	const message = getMessageByIndex({ messages, messageIndex });
	const parts: Exclude<ModelMessage["content"], string> =
		typeof message.content === "string"
			? [{ type: "text" as const, text: message.content }]
			: message.content;
	return parts.map((part, partIndex) => ({
		...part,
		id: getMessagePartIdentity({ messages, messageIndex, partIndex }),
	}));
}

export function attachIdsToMessages(
	messages: ModelMessage[],
): IdentifiableMessage[] {
	return messages.map((message, messageIndex) => ({
		id: getMessageIdentity({ messages, messageIndex }),
		parts: attachIdsToParts({ messages, messageIndex }),
		raw: message,
	}));
}

export function stripIdsFromMessages(
	messages: IdentifiableMessage[],
): ModelMessage[] {
	return messages.map((message) => ({
		...message.raw,
	}));
}

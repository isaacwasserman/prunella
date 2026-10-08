import { createHash } from "node:crypto";
import type { ModelMessage } from "ai";
import { urlAlphabet } from "nanoid";

type Part = Exclude<ModelMessage["content"], string>[number];

export type IdentifiablePart = Part & {
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

// Gives a deterministic nanoid-style ID: the same input always gives the same ID.
export function shortHashString(str: string, size = 8): string {
	const bytes = createHash("sha256").update(str).digest();
	let id = "";
	for (let i = 0; i < size; i++) {
		id += urlAlphabet[bytes[i]! & 63];
	}
	return id;
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

/** A message's parts; string content is one text part. */
function partsOf(message: ModelMessage): Part[] {
	return typeof message.content === "string"
		? [{ type: "text", text: message.content }]
		: message.content;
}

/**
 * The fields that identify a part. Provider fields are left out, and a tool
 * call or result is known by its call ID, so a message keeps its IDs when it
 * is stored and converted again.
 */
function canonicalPart(part: Part | null): unknown {
	if (!part) return null;
	switch (part.type) {
		case "text":
		case "reasoning":
			return { type: part.type, text: part.text };
		case "tool-call":
		case "tool-result":
			return {
				type: part.type,
				toolCallId: part.toolCallId,
				toolName: part.toolName,
			};
		default: {
			const {
				providerOptions: _,
				providerMetadata: __,
				...content
			} = part as Part & {
				providerOptions?: unknown;
				providerMetadata?: unknown;
			};
			return content;
		}
	}
}

function canonicalMessage(message: ModelMessage | null): unknown {
	if (!message) return null;
	return { role: message.role, parts: partsOf(message).map(canonicalPart) };
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
	return hashString(
		JSON.stringify(canonicalMessage(previousMessage)) +
			JSON.stringify(canonicalMessage(targetMessage)),
	);
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
	const parts = partsOf(getMessageByIndex({ messages, messageIndex }));
	return shortHashString(
		JSON.stringify(canonicalMessage(previousMessage)) +
			JSON.stringify(canonicalPart(parts[partIndex - 1] ?? null)) +
			JSON.stringify(canonicalPart(parts[partIndex] ?? null)),
	);
}

export function attachIdsToParts({
	messages,
	messageIndex,
}: { messages: ModelMessage[]; messageIndex: number }): IdentifiablePart[] {
	const parts = partsOf(getMessageByIndex({ messages, messageIndex }));
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

/** Tokens of the parts after a part, by message and part index. */
export type TokensAfter = (messageIndex: number, partIndex: number) => number;

export function tokensAfterParts(
	messages: IdentifiableMessage[],
	sizeOf: (part: IdentifiablePart) => number,
): TokensAfter {
	const after = messages.map((message) => message.parts.map(() => 0));
	let total = 0;
	for (let mi = messages.length - 1; mi >= 0; mi--) {
		const parts = messages[mi]!.parts;
		for (let pi = parts.length - 1; pi >= 0; pi--) {
			after[mi]![pi] = total;
			total += sizeOf(parts[pi]!);
		}
	}
	return (messageIndex, partIndex) => after[messageIndex]?.[partIndex] ?? 0;
}

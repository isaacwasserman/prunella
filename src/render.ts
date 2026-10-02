import type { ModelMessage } from "ai";
import {
	type CompactorSummary,
	type PartSpan,
	getPartIdsInSpan,
	summaryToMessage,
} from "./compaction";
import type { IdentifiableMessage } from "./utils";

const RECALL_TOOL_NAME = "recall-pruned";

type RenderedPart = Exclude<ModelMessage["content"], string>[number];

/**
 * A tool call or result keeps its type, so tool messages stay valid and every
 * result still has its call.
 */
export function createPlaceholder(
	partId: string,
	originalPart: IdentifiableMessage["parts"][number],
): RenderedPart {
	const text = `This part of the message has been pruned for token efficiency. To have its content revealed, run the ${RECALL_TOOL_NAME} tool with pruneId "${partId}".`;
	const { id: _, ...rawPart } = originalPart;
	if (rawPart.type === "tool-result") {
		return { ...rawPart, output: { type: "text", value: text } };
	}
	if (rawPart.type === "tool-call") {
		return { ...rawPart, input: { pruned: text } };
	}
	return { type: "text", text };
}

function toMessage(raw: ModelMessage, parts: RenderedPart[]): ModelMessage {
	if (raw.role === "system") {
		return {
			...raw,
			content: parts
				.map((part) => (part.type === "text" ? part.text : ""))
				.join("\n"),
		};
	}
	return { ...raw, content: parts } as ModelMessage;
}

export function renderMessages({
	messages,
	mask,
	summaries,
}: {
	messages: IdentifiableMessage[];
	mask: Set<string>;
	summaries: CompactorSummary[];
}): ModelMessage[] {
	const summaryByFirstPartId = new Map<string, CompactorSummary>();
	const coveredByCompaction = new Set<string>();
	for (const summary of summaries) {
		for (const span of summary.spans) {
			summaryByFirstPartId.set(span.firstPartId, summary);
			for (const id of getPartIdsInSpan({ span, messages })) {
				coveredByCompaction.add(id);
			}
		}
	}

	const result: ModelMessage[] = [];
	const emittedSummaries = new Set<string>();

	for (const message of messages) {
		const unchanged = message.parts.every(
			(part) =>
				!summaryByFirstPartId.has(part.id) &&
				!coveredByCompaction.has(part.id) &&
				!mask.has(part.id),
		);
		if (unchanged) {
			result.push(message.raw);
			continue;
		}

		const outputParts: RenderedPart[] = [];

		for (const part of message.parts) {
			const summary = summaryByFirstPartId.get(part.id);
			if (summary && !emittedSummaries.has(summary.id)) {
				emittedSummaries.add(summary.id);
				if (outputParts.length > 0) {
					result.push(toMessage(message.raw, outputParts.splice(0)));
				}
				result.push(summaryToMessage(summary));
				continue;
			}

			if (coveredByCompaction.has(part.id)) {
				continue;
			}

			if (mask.has(part.id)) {
				outputParts.push(createPlaceholder(part.id, part));
				continue;
			}

			const { id: _, ...rawPart } = part;
			outputParts.push(rawPart);
		}

		if (outputParts.length > 0) {
			result.push(toMessage(message.raw, outputParts));
		}
	}

	return result;
}

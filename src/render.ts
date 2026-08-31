import type { ModelMessage, TextPart } from "ai";
import {
	type CompactorSummary,
	type PartSpan,
	getPartIdsInSpan,
	summaryToMessage,
} from "./compaction";
import type { IdentifiableMessage } from "./utils";

const RECALL_TOOL_NAME = "recall-pruned";

function createPlaceholder(
	partId: string,
	originalPart: IdentifiableMessage["parts"][number],
): TextPart {
	return {
		type: "text",
		text: `This part of the message has been pruned for token efficiency. To have its content revealed, run the ${RECALL_TOOL_NAME} tool with pruneId "${partId}".`,
	};
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
	const activeSummaries = summaries.filter((s) => !s.collapsed);

	const summaryByFirstPartId = new Map<string, CompactorSummary>();
	const coveredByCompaction = new Set<string>();
	for (const summary of activeSummaries) {
		summaryByFirstPartId.set(summary.firstPartId, summary);
		for (const id of getPartIdsInSpan({
			span: {
				firstPartId: summary.firstPartId,
				lastPartId: summary.lastPartId,
			},
			messages,
		})) {
			coveredByCompaction.add(id);
		}
	}

	const result: ModelMessage[] = [];

	for (const message of messages) {
		const outputParts: ModelMessage["content"][number][] = [];

		for (const part of message.parts) {
			const summary = summaryByFirstPartId.get(part.id);
			if (summary) {
				if (outputParts.length > 0) {
					result.push({
						...message.raw,
						content: outputParts.splice(0),
					} as ModelMessage);
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
			result.push({
				...message.raw,
				content: outputParts,
			} as ModelMessage);
		}
	}

	return result;
}

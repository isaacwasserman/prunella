import type { LanguageModel, ModelMessage } from "ai";
import { estimateTokenCount } from "tokenx";
import {
	type CompactionOptions,
	Compactor,
	type CompactorStore,
	type CompactorSummary,
	getPartIdsInSpan,
} from "./compaction";
import type { CompactorHooks } from "./hooks";
import { type PressureMeasure, Pruner, type PruningPolicy } from "./pruning";
import { renderMessages, renderedPartTokens } from "./render";
import type { RuntimeConfig } from "./runtime-config";
import {
	type IdentifiableMessage,
	attachIdsToMessages,
	partTokens,
} from "./utils";

/** Measure the conversation the way compaction does: rendered, with summaries in place. */
function measureRendered({
	messagesWithIds,
	existingSummaries,
}: {
	messagesWithIds: IdentifiableMessage[];
	existingSummaries: CompactorSummary[];
}): PressureMeasure {
	const parts = new Map(
		messagesWithIds.flatMap((message) =>
			message.parts.map((part) => [part.id, part] as const),
		),
	);
	const covered = new Set(
		existingSummaries.flatMap((summary) =>
			summary.spans.flatMap((span) =>
				getPartIdsInSpan({ span, messages: messagesWithIds }),
			),
		),
	);
	return {
		size: (mask) =>
			estimateTokenCount(
				JSON.stringify(
					renderMessages({
						messages: messagesWithIds,
						mask,
						summaries: existingSummaries,
					}),
				),
			),
		savings: (partId) => {
			const part = parts.get(partId);
			if (!part || covered.has(partId)) return 0;
			return partTokens(part) - renderedPartTokens(part, true);
		},
	};
}

export class Prunella<TRuntimeConfig extends RuntimeConfig = undefined> {
	private pruner: Pruner;
	private compactor: Compactor<TRuntimeConfig> | undefined;

	constructor(args: {
		pruningPolicy: PruningPolicy;
		compaction?: {
			enabled: true;
			store: CompactorStore<TRuntimeConfig>;
			model: LanguageModel;
			policy: CompactionOptions;
			summaryPrompt?: string;
			hooks?: CompactorHooks<TRuntimeConfig>;
		};
	}) {
		this.pruner = new Pruner({
			pruningPolicy: args.pruningPolicy,
		});
		this.compactor = args.compaction
			? new Compactor<TRuntimeConfig>({
					store: args.compaction.store,
					model: args.compaction.model,
					options: args.compaction.policy,
					summaryPrompt: args.compaction.summaryPrompt,
					hooks: args.compaction.hooks,
				})
			: undefined;
	}

	public async prepare({
		messages,
		sessionId,
		config,
	}: { messages: ModelMessage[]; sessionId: string; config: TRuntimeConfig }) {
		const messagesWithIds = attachIdsToMessages(messages);
		const existingSummaries = this.compactor
			? await this.compactor.loadSummaries({
					messagesWithIds,
					sessionId,
					config,
				})
			: [];

		const { mask, tools: pruningTools } = this.pruner.prepare({
			messages: messagesWithIds,
			measure: measureRendered({ messagesWithIds, existingSummaries }),
		});

		const compaction = this.compactor
			? await this.compactor.prepare({
					messages,
					messagesWithIds,
					mask,
					existingSummaries,
					sessionId,
					config,
				})
			: undefined;
		const summaries = compaction?.summaries ?? [];

		const rendered = renderMessages({
			messages: messagesWithIds,
			mask,
			summaries,
		});

		return {
			messages: rendered,
			/** `recall-pruned` when a part was pruned, and the compactor tools when summaries exist. */
			tools: {
				...(mask.size > 0 ? pruningTools : {}),
				...(compaction && summaries.length > 0 ? compaction.tools : {}),
			},
		};
	}
}

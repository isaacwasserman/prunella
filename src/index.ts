import type { UsageEstimator } from "@tokenxl/count";
import type { LanguageModel, ModelMessage } from "ai";
import {
	type CompactionOptions,
	Compactor,
	type CompactorStore,
	type CompactorSummary,
	getPartIdsInSpan,
} from "./compaction";
import type { CompactorHooks } from "./hooks";
import { type PressureMeasure, Pruner, type PruningPolicy } from "./pruning";
import { measureParts, renderMessages } from "./render";
import type { RuntimeConfig } from "./runtime-config";
import { type IdentifiableMessage, attachIdsToMessages } from "./utils";

export type {
	CompactionOptions,
	CompactorStore,
	CompactorSummary,
	PartSpan,
} from "./compaction";
export type { CompactorHooks } from "./hooks";
export type { PartAge, PressureCondition, PruningPolicy } from "./pruning";
export type { RuntimeConfig } from "./runtime-config";

/** Measure the conversation the way compaction does: rendered, with summaries in place. */
function measureRendered({
	estimator,
	messagesWithIds,
	existingSummaries,
}: {
	estimator: UsageEstimator;
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
	const partSize = measureParts(estimator, messagesWithIds);
	return {
		size: (mask) =>
			estimator.count({
				messages: renderMessages({
					messages: messagesWithIds,
					mask,
					summaries: existingSummaries,
				}),
			}),
		savings: (partId) => {
			const part = parts.get(partId);
			if (!part || covered.has(partId)) return 0;
			return partSize(part, false) - partSize(part, true);
		},
	};
}

export class Prunella<TRuntimeConfig extends RuntimeConfig = undefined> {
	private pruner: Pruner;
	private compactor: Compactor<TRuntimeConfig> | undefined;
	private estimator: UsageEstimator;

	constructor(args: {
		/**
		 * Counts tokens, for example `createUsageEstimator("anthropic/claude-sonnet-5.5")`
		 * from `@tokenxl/count`. Use a profile for the model that gets the messages.
		 */
		estimator: UsageEstimator;
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
		this.estimator = args.estimator;
		this.pruner = new Pruner({
			pruningPolicy: args.pruningPolicy,
			estimator: args.estimator,
		});
		this.compactor = args.compaction
			? new Compactor<TRuntimeConfig>({
					store: args.compaction.store,
					model: args.compaction.model,
					options: args.compaction.policy,
					summaryPrompt: args.compaction.summaryPrompt,
					hooks: args.compaction.hooks,
					estimator: args.estimator,
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

		const pruneWith = (summaries: CompactorSummary[]) =>
			this.pruner.prepare({
				messages: messagesWithIds,
				measure: measureRendered({
					estimator: this.estimator,
					messagesWithIds,
					existingSummaries: summaries,
				}),
			});
		let { mask, tools: pruningTools } = pruneWith(existingSummaries);

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
		// Each new or merged summary gets a new ID, so the IDs show a change.
		const summariesChanged =
			summaries.map((summary) => summary.id).join() !==
			existingSummaries.map((summary) => summary.id).join();
		if (summariesChanged) {
			// Choose pressure parts again, so parts that a summary now covers do not cause more pruning.
			({ mask, tools: pruningTools } = pruneWith(summaries));
		}

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

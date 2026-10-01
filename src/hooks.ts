import type { ModelMessage } from "ai";
import type { CompactorSummary } from "./compaction";
import type { RuntimeConfig } from "./runtime-config";

export type CompactorHooks<TRuntimeConfig extends RuntimeConfig = undefined> = {
	/**
	 * Fires only when compaction is about to summarize or merge, not on every
	 * `prepare`. Return `false` to skip compaction for this `prepare`; existing
	 * summaries still apply.
	 */
	onCompactStart?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		messages: ModelMessage[];
		existingSummaries: CompactorSummary[];
		estimatedTokens: number;
	}) => Promise<void> | Promise<boolean>;

	/** Fires once compaction has finished, if `onCompactStart` let it start. */
	onCompactEnd?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		summaries: CompactorSummary[];
		estimatedTokens: number;
		iterations: number;
		summariesCreated: number;
		summariesMerged: number;
	}) => Promise<void>;

	/**
	 * Fires when a summary is written. The summary replaces every earlier
	 * summary of the session, listed in `replacedSummaries`.
	 */
	onSummaryCreate?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		summary: CompactorSummary;
		replacedSummaries: CompactorSummary[];
	}) => Promise<void>;

	/** Fires when several summaries, written before summaries were rolling, are merged into one. */
	onSummaryMerge?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		mergedSummary: CompactorSummary;
		sourceSummaries: CompactorSummary[];
	}) => Promise<void>;
};

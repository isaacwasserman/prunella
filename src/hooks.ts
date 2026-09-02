import type { ModelMessage } from "ai";
import type { CompactorSummary } from "./compaction";
import type { RuntimeConfig } from "./runtime-config";

export type CompactorHooks<TRuntimeConfig extends RuntimeConfig = undefined> = {
	onCompactStart?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		messages: ModelMessage[];
		existingSummaries: CompactorSummary[];
		estimatedTokens: number;
	}) => Promise<void>;

	onCompactEnd?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		summaries: CompactorSummary[];
		estimatedTokens: number;
		iterations: number;
	}) => Promise<void>;

	onSummaryCreate?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		summary: CompactorSummary;
	}) => Promise<void>;

	onSummaryMerge?: (params: {
		config: TRuntimeConfig;
		sessionId: string;
		mergedSummary: CompactorSummary;
		sourceSummaries: [CompactorSummary, CompactorSummary];
	}) => Promise<void>;
};

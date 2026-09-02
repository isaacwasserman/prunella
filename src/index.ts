import type { LanguageModel, ModelMessage } from "ai";
import {
	type CompactionOptions,
	Compactor,
	type CompactorStore,
} from "./compaction";
import { Pruner, type PruningPolicy } from "./pruning";
import { renderMessages } from "./render";
import type { RuntimeConfig } from "./runtime-config";
import { attachIdsToMessages } from "./utils";

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
				})
			: undefined;
	}

	public async prepare({
		messages,
		sessionId,
		config,
	}: { messages: ModelMessage[]; sessionId: string; config: TRuntimeConfig }) {
		const messagesWithIds = attachIdsToMessages(messages);

		const { mask, tools } = this.pruner.prepare({
			messages: messagesWithIds,
		});

		const { summaries } = this.compactor
			? await this.compactor.prepare({
					messages,
					sessionId: sessionId,
					config,
				})
			: { summaries: [] as import("./compaction").CompactorSummary[] };

		const rendered = renderMessages({
			messages: messagesWithIds,
			mask,
			summaries,
		});

		return { messages: rendered, tools };
	}
}

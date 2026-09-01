import type { LanguageModel, ModelMessage } from "ai";
import {
	type CompactionOptions,
	Compactor,
	type CompactorStore,
} from "./compaction";
import { Pruner, type PruningPolicy } from "./pruning";
import { renderMessages } from "./render";
import { attachIdsToMessages } from "./utils";

export class Prunella {
	private pruner: Pruner;
	private compactor: Compactor | undefined;

	constructor(args: {
		pruningPolicy: PruningPolicy;
		compaction?: {
			enabled: true;
			store: CompactorStore;
			model: LanguageModel;
			policy: CompactionOptions;
			summaryPrompt?: string;
		};
	}) {
		this.pruner = new Pruner({
			pruningPolicy: args.pruningPolicy,
		});
		this.compactor = args.compaction
			? new Compactor({
					store: args.compaction.store,
					model: args.compaction.model,
					policy: args.compaction.policy,
				})
			: undefined;
	}

	public async prepare({
		messages,
		sessionId,
	}: { messages: ModelMessage[]; sessionId?: string }) {
		const messagesWithIds = attachIdsToMessages(messages);

		const { mask, tools } = this.pruner.prepare({
			messages: messagesWithIds,
		});

		const { summaries } = this.compactor
			? await this.compactor.prepare({
					messages,
					sessionId: sessionId!,
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

using LLama.Native;

using Llamba.Server;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Llamba.Batching {
	/// <summary>
	/// <b>Helper class for <see cref="BatchProcessor"/> with sole responsibility of refreshing the context by bringing 'queued' requests to the list of 'active' requests. </b> Contains two implementations:
	/// <para>**************************************************************************************************************************************************************************************************************</para>
	/// <para><b>(1) Full context refresh that re-digests the prompt of ALL requests (even existing ones), each time a new request is added, offering better T/s speeds in expense of bigger waiting time.</b></para>
	/// <para><i>- The advantage of this is that some tokens will share their position in multiple sequences, causing inference to be quicker because they're reused.</i></para>
	/// <para><i>- The disadvantage is that queued requests will have to wait until a number of 'active' requests have been finished, and the refresh will happen..</i></para>
	/// <para><b>(2) Additive context refresh that defrags the cache and appends new requests each time new requests are added, allowing quicker and more frequent addressing of pending requests.</b></para>
	/// <para><i>- The advantage of this is that there'd be no significant drawbacks is new requests were to be prompt-processed immediately after they were received.</i></para>
	/// <para><i>- The disadvantage is that because tokens aren't shared in as many sequences as possible, the T/s of contexts generated with this method will be a little lower..</i></para>
	/// <para>**************************************************************************************************************************************************************************************************************</para>
	/// <para>*** TLDR: <b>Full Context Refresh</b> is better for data generation, cleaning, etc (because of bigger T/s), and <b>Additive Context Refresh</b> might be better for servers (because of smaller waiting times) ***</para>
	/// </summary>
	public static class ContextRefresher {
		/// <summary> The maximum amount of context that should be full before the context is deemed valid to be refreshed. The bigger this value, the more frequent the refreshes. </summary>
		/// <remarks> You may increase this when running a server to make new pending requests get in the context quicker. <b>(WARNING: FullRefreshing often will have speed drawbacks).</b></remarks>
		public static float contextPercentSweetspot = 0.5f; // Final reminder that this SHOULDN'T be too high unless you do `AdditiveContextRefresh` for low latency responses to multiple clients.

		/// <summary> When an update is needed, clears the cache completely and rebuilds it with the new requests included. </summary>
		/// <remarks> Internally decodes the newly added prompts, leaving only the last token of their sequence undecoded. </remarks>
		public static void FullContextRefresh(SafeLLamaContextHandle context, LLamaBatch batch, int maxContextSize, int maxBatchSize, List<InferenceRequest> active, ConcurrentQueue<InferenceRequest> queued) {
			if (!ShouldRefresh(context, maxContextSize, active, queued, out var currentContextSize)) { return; } // If there's no reason to make any changes to the context, just keep going.
			NativeApi.llama_kv_cache_clear(context); // Clear the cache FULLY -- before we basically rebuild it from scratch.
			UnqueueRequests(maxContextSize, active, queued, currentContextSize); // Fill up the 'active' list with 'queued' inference requests.
			PromptProcessNewRequests(context, batch, active, maxBatchSize, 0);	 // Process the prompt for both existing and new requests.
		}

		/// <summary> When an update is needed, defrags the cache and appends new requests as new sequences. </summary>
		/// <remarks> Internally decodes the newly added prompts, leaving only the last token of their sequence undecoded. </remarks>
		public static void AdditiveContextRefresh(SafeLLamaContextHandle context, LLamaBatch batch, int maxContextSize, int maxBatchSize, List<InferenceRequest> active, ConcurrentQueue<InferenceRequest> queued) {
			if (!ShouldRefresh(context, maxContextSize, active, queued, out var currentContextSize)) { return; } // If there's no reason to make any changes to the context, just keep going.
			PartialContextRefreshHelper.DefragContext(context, active); // Defrag the current context, bringing remaining sequences to have sequence IDs of [0,1,2,...,n]
			int newRequestCount = UnqueueRequests(maxContextSize, active, queued, currentContextSize);		// Fill up the 'active' list with 'queued' inference requests.
			PromptProcessNewRequests(context, batch, active, maxBatchSize, active.Count - newRequestCount); // Prompt process the new requests, and prepare them for inference.
		}


		static bool ShouldRefresh(SafeLLamaContextHandle context, int maxContextSize, List<InferenceRequest> active, ConcurrentQueue<InferenceRequest> queued, out int currentContextSize) {
			void Remove(int i) { context.ClearSeq(active[i].sequenceID); active.RemoveAt(i); } // Deactivate and optionally clear the sequence in the context. Leaving it is slightly faster.
			currentContextSize = 0; // Go through all the requests and if they're no longer active, remove them -- otherwise add their total count to the total, so we'll know if we need to refresh the actives.
			for (int i = active.Count - 1; i >= 0; i--) { if (!active[i].needsGen) { Remove(i); } else { currentContextSize += active[i].totalTokens; } }
			return !(queued.IsEmpty || currentContextSize >= contextPercentSweetspot * maxContextSize); // If the cache is sufficiently loaded, we can let it be and continue inferencing for the existing sequences.
		}
		static int UnqueueRequests(int maxContextSize, List<InferenceRequest> active, ConcurrentQueue<InferenceRequest> queued, int currentContextSize) {
			var (awaitingTokenSize, minAwaitingSize, newRequestCount) = (0, int.MaxValue, 0); // Get a bunch of stats about the current cache.
			for (int i = active.Count - 1; i >= 0; i--) { var request = active[i]; awaitingTokenSize += request.remainingTokensCount; minAwaitingSize = Math.Min(request.remainingTokensCount, minAwaitingSize); }

			// Get the minimum amount of empty space we need to remain empty.
			var maxFillBunch = maxContextSize - active.Count * minAwaitingSize;

			// Prepare queued requests for becoming active, based on the available space in the context
			while (currentContextSize < maxFillBunch) {
				if (!queued.TryPeek(out var peekedRequest)) { break; } // Peek the next-in-queue.
				if (currentContextSize + peekedRequest.inputTokensCount + peekedRequest.remainingTokensCount > maxFillBunch) { break; }
				if (!queued.TryDequeue(out var newRequest)) { break; } // Shouldn't ever happen tho.

				// Once the new request is accepted, add it to the list and update our variables.
				active.Add(newRequest);
				currentContextSize += newRequest.inputTokensCount;
				awaitingTokenSize += newRequest.inputTokensCount;
				minAwaitingSize = Math.Min(newRequest.remainingTokensCount, minAwaitingSize);
				maxFillBunch = maxContextSize - (active.Count * minAwaitingSize);
				newRequestCount++;
			}
			return newRequestCount;
		}
		static void PromptProcessNewRequests(SafeLLamaContextHandle context, LLamaBatch batch, List<InferenceRequest> active, int maxBatchSize, int startCount) {
			for (int i = startCount; i < active.Count; i++) {
				var newRequest = active[i];
				var seqID = (LLamaSeqId) (newRequest.sequenceID = i);

				// Process all except the last token, which will be used for inference later.
				for (int j = 0; j < newRequest.promptTokens.Count - 1; j++) { // NOT the last token, coz we'll decode it with logits.
					batch.Add(newRequest.promptTokens[j], pos: j, sequence: seqID, logits: false);	// We do not want logits for any of these.
					if (batch.TokenCount == maxBatchSize) { context.Decode(batch); batch.Clear(); } // Decode and clear the batch if it's full.
				}
			}
		}


		static class PartialContextRefreshHelper {
			static HashSet<int> occupiedSeqIDs = [];
			static List<int> freeSeqIDs = [];
			static List<InferenceRequest> shouldMove = [];

			/// <summary> Defragments the context, putting everything in a tight sequence -- effectively reducing the final length of the sequence. </summary>
			public static void DefragContext(SafeLLamaContextHandle context, List<InferenceRequest> active) {
				occupiedSeqIDs.Clear(); freeSeqIDs.Clear(); shouldMove.Clear(); // Clear buffers, and rebuild them, gathering free sequence IDs and requests we should defrag.
				for (int i = 0; i < active.Count; i++) { var req = active[i]; occupiedSeqIDs.Add(req.sequenceID); if (req.sequenceID >= active.Count) { shouldMove.Add(req); } }
				for (int i = 0; i < active.Count; i++) { if (!occupiedSeqIDs.Contains(i)) { freeSeqIDs.Add(i); } } // Finally, defrag by moving the requests to the first available sequence.
				for (int i = 0; i < shouldMove.Count; i++) { var newSeqID = freeSeqIDs[0]; context.MoveSeq(shouldMove[i].sequenceID, shouldMove[i].sequenceID = newSeqID); freeSeqIDs.RemoveAt(0); }
			}
		}
	}
}

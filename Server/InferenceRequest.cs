using Llamba.Sampling;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Llamba.Server {
	/// <summary> A partial response to a request.. A bunch of those are sent back to the client for a single request, each containing (usually) a single token when streaming. </summary>
	/// <remarks> NOTE: Even a streaming response may be consisted of multiple tokens in cases of multi-token (non-english) characters -- e.g.: emojis, chinese, greek, etc. </remarks>
	public struct InferenceResponse(string response, int sampledToken, string stopReason, int tokensCount) {
		public string response = response;
		public int sampledToken = sampledToken;
		public string stopReason = stopReason;   // TODO: Move stopReason
		public bool stop = stopReason != "none"; // ..+ token counts
		public int tokensCount = tokensCount;    // .. to ChatEndpoint
	}

	/// <summary> A request for inference made from the client. Note that this is the internal representation and not the query. </summary>
	/// <remarks> It will remain active until the end_token is detected, or until the necessary amount of tokens were inferenced from the model.</remarks>
	public class InferenceRequest : IDisposable {
		public int remainingTokensCount => requestedTokensCount - receivedTokensCount;
		public int totalTokens => inputTokensCount + receivedTokensCount;
		public int receivedTokensCount => sampler.samplerParams.receivedTokensCount;
		public IReadOnlyList<int> promptTokens => sampler.samplerParams.promptTokens;

		// General info about the request
		public readonly int inputTokensCount;
		public readonly int requestedTokensCount;
		public readonly string initialPrompt;

		readonly static int end_token = Model.instance.eotID;


		// Exposed for the Processor
		public bool needsGen = true;
		public int sequenceID = -1;
		public int sampledToken { get; private set; } = -1;

		ISampler sampler;

		public ConcurrentQueue<InferenceResponse> nextResponse { get; } = new();

		public InferenceRequest(string prompt, ChatQuery query, IEnumerable<int> promptTokens, ISampler sampler) {
			(this.sampler = sampler).Initialize(query, promptTokens);
			initialPrompt = prompt;
			requestedTokensCount = query.max_tokens ?? 300;
			inputTokensCount = sampler.samplerParams.promptTokens.Count;
			sampledToken = sampler.samplerParams.promptTokens[^1];
		}

		int heldTokens = 0;

		/// <summary> Sample from the received logits into a request response for the client. </summary>
		public void HandleLogits(Span<float> logits) {
			sampledToken = sampler.SampleToken(logits);

			string stopReason = string.Empty; //TODO: Move to ChatEndpoint -- not related to logits.
			if (sampledToken == end_token) { needsGen = false; stopReason = "stop"; }
			else if (sampler.samplerParams.receivedTokensCount >= requestedTokensCount) { needsGen = false; stopReason = "limit"; }

			if (needsGen) { sampler.PostSample(sampledToken); }

			var sampledText = Model.vocab[sampledToken]; // In case of non-character tokens (e.g.: emoji parts), we'll wait until a complete character is made.
			if ((heldTokens > 0 || (sampledText.Length != 0 && sampledText[^1] == '�')) && !DecodingManager.Decode(this, out sampledText, sampledToken)) { heldTokens++; return; }

			nextResponse.Enqueue(new() {
				response = stopReason == "stop" ? "" : sampledText,
				sampledToken = sampledToken,
				stopReason = stopReason,
				stop = needsGen,
				tokensCount = heldTokens + 1,
			});
			heldTokens = 0;
		}

		// Dispose is automatically called after ChatEndpoint finishes `using` it.
		async void IDisposable.Dispose() {
			needsGen = false;
			sampler.samplerParams.Dispose();
			DecodingManager.ReturnDecoder(this);
		}
	}
}
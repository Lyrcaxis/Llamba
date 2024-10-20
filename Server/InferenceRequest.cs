using Llamba.Sampling;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;

namespace Llamba.Server {
    /// <summary> A partial response to a request.. A bunch of those are sent back to the client for a single request, each containing (usually) a single token when streaming. </summary>
    /// <remarks> NOTE: Even a streaming response may be consisted of multiple tokens in cases of multi-token (non-english) characters -- e.g.: emojis, chinese, greek, etc. </remarks>
    public struct InferenceResponse(string response, string stopReason, int tokensCount) {
        public string response = response;
        public string stopReason = stopReason;   // TODO: Move stopReason
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

        readonly static int eot_token_ID = Model.instance.eotID;


        // Exposed for the Processor
        public bool needsGen = true;
        public int sequenceID = -1;
        public int sampledToken { get; private set; } = -1;

        ISampler sampler;
        HashSet<int> stopTokens;
        bool hasStopped;

        string totalPrompt;

        public ConcurrentQueue<InferenceResponse> nextResponse { get; } = new();

        public InferenceRequest(string prompt, IQueryParamsContainer query, IEnumerable<int> promptTokens, ISampler sampler, HashSet<int> stopTokens) {
            (this.sampler = sampler).Initialize(query, promptTokens);
            this.stopTokens = stopTokens;
            totalPrompt = initialPrompt = prompt;
            requestedTokensCount = query.max_tokens ?? 300;
            inputTokensCount = sampler.samplerParams.promptTokens.Count;
            sampledToken = sampler.samplerParams.promptTokens[^1];
        }

        int heldTokenCount = 0; // The amount of non-character tokens that have been held for future decoding.

        /// <summary> Sample from the received logits into a request response for the client. </summary>
        public void HandleLogits(Span<float> logits) {
            // Sample the token and stop instantly if it's a EOT token.
            if ((sampledToken = sampler.SampleToken(logits)) == eot_token_ID) { SendClosingResponse("stop"); return; }

            // In case of non-character tokens (e.g.: emoji parts), we'll wait until a complete character is made.
            var sampledText = Model.vocab[sampledToken];
            if ((heldTokenCount > 0 || (sampledText.Length != 0 && sampledText[^1] == '�'))) {
                if (!DecodingManager.Decode(this, out sampledText, sampledToken)) { heldTokenCount++; }
                else { heldTokenCount = 0; }
            }

            // If the sampled text is unicode, enqueue the response to the queue and reset the counters.
            if (heldTokenCount == 0) { nextResponse.Enqueue(new() { response = sampledText, stopReason = "none", tokensCount = heldTokenCount + 1 }); }
            if (stopTokens?.Contains(sampledToken) == true) { SendClosingResponse("stop"); return; }


            // Finally, check if we should terminate the inference for this request because the max_token limit was hit.
            if (sampler.samplerParams.receivedTokensCount + 1 >= requestedTokensCount) { SendClosingResponse("limit"); }
            else { sampler.PostSample(sampledToken); } // .. otherwise pass the sampled token to the sampler to update its cache. This will never trigger if the request no longer needs gen.


            // Sends an end-of-response response, with an empty response and a stop reason.
            void SendClosingResponse(string stopReason) { needsGen = false; nextResponse.Enqueue(new() { response = "", stopReason = stopReason, tokensCount = 0 }); }
        }

        // Dispose is automatically called after ChatEndpoint finishes `using` it.
        async void IDisposable.Dispose() {
            needsGen = false;
            sampler.samplerParams.Dispose();
            DecodingManager.ReturnDecoder(this);
        }
    }
}
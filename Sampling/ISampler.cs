using System;
using System.Collections.Generic;

namespace Llamba.Sampling {
	/// <summary> Base interface for all sampling pipelines. Contains a method to initialize, one to sample, and another to update the sampling parameters. </summary>
	/// <remarks> We typically use one sampler instance <b>PER REQUEST</b>, so try to keep this lightweight and apply pooling when possible. </remarks>
	public interface ISampler {
		RequestSamplingParams samplerParams { get; set; }

		/// <summary> Performs the sampling operations and returns the ID of the chosen token. </summary>
		int SampleToken(Span<float> logits);

		/// <summary> Updates the sampler params based on the token that was sampled. This method should typically stay unchanged. </summary>
		void PostSample(int sampledToken) {
			samplerParams.receivedTokensCount++;
			samplerParams.promptTokens.Add(sampledToken);
			PostSampleInternal(sampledToken);
		}

		/// <summary> This is the method to override from implementing classes to handle post-sample operations like repetition tracking etc. </summary>
		void PostSampleInternal(int sampledToken) { }

		/// <summary> Initializes the sampler for the given query. </summary>
		void Initialize(ChatQuery query, IEnumerable<int> promptTokens);
	}

	/// <summary> The per-request information required for sampling. </summary>
	public class RequestSamplingParams(IEnumerable<int> promptTokens) : IDisposable {
		public int receivedTokensCount { get; set; }
		public List<int> promptTokens { get; init; } = [.. promptTokens];

		async public virtual void Dispose() { }
	}
}

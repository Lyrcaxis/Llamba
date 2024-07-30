using System;
using System.Collections.Generic;
using System.Numerics.Tensors;

namespace Llamba.Sampling {
	/// <summary> Sampler that's quick but not as in-depth as the LLamaSampler. Effective for achieving high speeds during batching. </summary>
	/// <remarks> The main difference is that the operations this sampler performs are all in C#, so there's no need for back-and-forth with the llama.cpp backend. </remarks>
	public class QuickSampler : ISampler {
		/// <summary> Toggle this to true if you don't want any newline tokens to be permitted as the model's output. </summary>
		public bool singleLine { get; set; } = false;

		/// <summary> Toggle this to true if you don't care about finding out what your model can or cannot do. </summary>
		public bool preventRefusals { get; set; } = false;

		/// <summary> Toggle this to true if you don't want ANY caps in your model's output. All-lowercase. </summary>
		public bool banCaps { get; set; } = false;

		/// <summary> The ideal response length, in tokens (0 = disabled). This will penalize logits for end-of-turn token based on the current response length. </summary>
		public int idealResponseLength { get; set; } = 0;

		/// <summary> Whether repetition should be penalized or not. !!! NOTE: WIP !!! </summary>
		public bool penalizeRepetition { get; set; } = false;

		public RequestSamplingParams samplerParams { get; set; }
		QuickSamplerRequestParams _samplerParams => samplerParams as QuickSamplerRequestParams;

		int ISampler.SampleToken(Span<float> logits) {
			// Flat-out prevent some tokens from being outputted, if specified.
			if (preventRefusals && samplerParams.receivedTokensCount < 10) { SmartBuffer.PreventRefusals(logits); }
			if (singleLine) { SmartBuffer.PreventNewlines(logits); }
			if (banCaps) { SmartBuffer.PreventCaps(logits); }
			SmartBuffer.PreventBadTokens(logits);

			// Alter the logits accordingly to attempt and meet ideal response length, if specified.
			if (idealResponseLength > 0) {
				var reducedLogits = -5 * (1 - Math.Min(samplerParams.receivedTokensCount, idealResponseLength) / (float) idealResponseLength);
				logits[Model.instance.eotID] = Math.Min(logits[Model.instance.eotID], reducedLogits);
			}

			// Penalize repetition by adding the penalty buffers to our logits before sampling.
			if (penalizeRepetition) {
				TensorPrimitives.Add(logits, _samplerParams.frequencyPenaltyBuffer, logits);
				TensorPrimitives.Add(logits, _samplerParams.presencePenaltyBuffer, logits);
			}

			// Add the direct logit biases to the logits.
			if (_samplerParams.logitBiasBuffer != null) { TensorPrimitives.Add(logits, _samplerParams.logitBiasBuffer, logits); }

			// Finally, sample the most-probable prediction after applying some randomization.
			return TensorPrimitives.IndexOfMax(SmartBuffer.ApplyTemperature(logits, 1f));
		}
		void ISampler.PostSampleInternal(int sampledToken) {
			if (!penalizeRepetition) { return; }

			// Register new token to the penalty buffers.
			_samplerParams.frequencyPenaltyBuffer[sampledToken] -= 0.1f;
			if (_samplerParams.presencePenaltyBuffer[sampledToken] == 0) { _samplerParams.presencePenaltyBuffer[sampledToken] -= 0.2f; }
		}

		void ISampler.Initialize(ChatQuery query, IEnumerable<int> promptTokens) {
			samplerParams = new QuickSamplerRequestParams(promptTokens);

			if (query.logit_bias != null) {
				_samplerParams.logitBiasBuffer = SmartBuffer.Rent();
				foreach (var bias_entry in query.logit_bias) { _samplerParams.logitBiasBuffer[bias_entry.Key] = bias_entry.Value; }
			}

			if (penalizeRepetition) {
				_samplerParams.presencePenaltyBuffer = SmartBuffer.Rent();
				_samplerParams.frequencyPenaltyBuffer = SmartBuffer.Rent();

				var inputTokensCount = samplerParams.promptTokens.Count;
				for (int i = inputTokensCount - 1; i > Math.Max(inputTokensCount - 1024, 1); i--) {
					var tokenToPenalize = _samplerParams.promptTokens[i];
					_samplerParams.frequencyPenaltyBuffer[tokenToPenalize] -= 0.2f * ((i / (2 * (float) inputTokensCount)));
					if (_samplerParams.presencePenaltyBuffer[tokenToPenalize] == 0) { _samplerParams.presencePenaltyBuffer[tokenToPenalize] -= 0.2f; }
				}
			}
		}

		class QuickSamplerRequestParams : RequestSamplingParams {
			public float[] presencePenaltyBuffer { get; set; }
			public float[] frequencyPenaltyBuffer { get; set; }
			public float[] logitBiasBuffer { get; set; }

			public QuickSamplerRequestParams(IEnumerable<int> promptTokens) : base(promptTokens) { }

			async public override void Dispose() {
				base.Dispose();
				if (presencePenaltyBuffer != null) { SmartBuffer.Return(presencePenaltyBuffer); }
				if (presencePenaltyBuffer != null) { SmartBuffer.Return(frequencyPenaltyBuffer); }
				if (logitBiasBuffer != null) { SmartBuffer.Return(logitBiasBuffer); }
			}
		}

	}
}

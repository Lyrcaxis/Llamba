using LLama.Native;

using System;
using System.Collections.Generic;
using System.Linq;

namespace Llamba.Sampling {
	/// <summary> Standard sampler that contains all (basic) sampling features available in the base version. </summary>
	/// <remarks> It's noticeably slower than the QuickSampler (~20x slower), but that's negligible unless it's used for batching. </remarks>
	public class LLamaSampler : ISampler {
		public float temperature { get; set; } = 0.5f;
		public float repetition_penalty { get; set; } = 1.2f;
		public float presence_penalty { get; set; } = 0;
		public float frequency_penalty { get; set; } = 0;
		public int repetition_range { get; set; } = 1024;

		public float top_p { get; set; } = 1;
		public float min_p { get; set; } = 0;

		(LLamaToken, float)[] logit_bias { get; set; }

		LLamaToken[] repetitionBuffer;

		public RequestSamplingParams samplerParams { get; set; }

		int ISampler.SampleToken(Span<float> logits) {
			var mHandle = Model.instance.context.NativeHandle;
			var tHandle = LLamaTokenDataArray.Create(logits);
			var maxTokensForRep = Math.Min(repetitionBuffer.Length, samplerParams.promptTokens.Count);
			for (int i = 0; i < maxTokensForRep; i++) {
				var index = samplerParams.promptTokens.Count - 1 - i;
				repetitionBuffer[i] = samplerParams.promptTokens[index];
			}
			tHandle.RepetitionPenalty(mHandle, repetitionBuffer.AsSpan(0, maxTokensForRep), repetition_penalty, frequency_penalty, presence_penalty);
			if (logit_bias != null) { tHandle.OverwriteLogits(logit_bias); }
			if (top_p != 1) { tHandle.TopP(mHandle, top_p); }
			if (min_p != 0) { tHandle.MinP(mHandle, min_p); }
			tHandle.Temperature(mHandle, temperature);

			return (int) tHandle.SampleToken(mHandle);
		}

		void ISampler.Initialize(ChatQuery query, IEnumerable<int> promptTokens) {
			samplerParams = new RequestSamplingParams(promptTokens);
			if (query.temperature.HasValue) { temperature = query.temperature.Value; }
			if (query.presence_penalty.HasValue) { presence_penalty= query.presence_penalty.Value; }
			if (query.frequency_penalty.HasValue) { frequency_penalty = query.frequency_penalty.Value; }
			if (query.repetition_penalty.HasValue) { repetition_penalty = query.repetition_penalty.Value; }
			if (query.repetition_range.HasValue) { repetition_range = query.repetition_range.Value; }
			if (query.top_p.HasValue) { top_p = query.top_p.Value; }
			if (query.min_p.HasValue) { min_p = query.min_p.Value; }
			logit_bias = query.logit_bias?.Select(x => ((LLamaToken) x.Key, x.Value)).ToArray();
			repetitionBuffer = new LLamaToken[repetition_range];
		}
	}
}

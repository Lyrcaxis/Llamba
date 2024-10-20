using LLama.Native;

using System;
using System.Collections.Generic;
using System.Linq;

namespace Llamba.Sampling {
    /// <summary> Standard sampler that contains all (basic) sampling features available in the base version. </summary>
    /// <remarks> It's noticeably slower than the QuickSampler (~20x slower), but that's negligible unless it's used for batching. </remarks>
    public class LLamaSampler : ISampler {
        public float temperature { get; set; } = 1;
        public float repetition_penalty { get; set; } = 1;
        public float presence_penalty { get; set; } = 0;
        public float frequency_penalty { get; set; } = 0;
        public int repetition_range { get; set; } = 1024;

        public int minimum_tokens { get; set; } = 10;

        public int top_k { get; set; } = 100;
        public float top_p { get; set; } = 0.9f;
        public float min_p { get; set; } = 0.1f;
        public float typical_p { get; set; } = 1;
        public float tail_free_z { get; set; } = 1;

        LLamaLogitBias[] logit_bias { get; set; }

        public RequestSamplingParams samplerParams { get; set; }
        SafeLLamaSamplerChainHandle sampler;

        static LLamaTokenDataArray arrayCache;

        int ISampler.SampleToken(Span<float> logits) {
            if (samplerParams.receivedTokensCount < minimum_tokens) { SmartBuffer.PreventRefusals(logits); }
            arrayCache = LLamaTokenDataArray.Create(logits);
            using var _ = LLamaTokenDataArrayNative.Create(arrayCache, out var cur_p);
            sampler.Apply(ref cur_p);
            var token = cur_p.Data[(int) cur_p.Selected].ID;
            sampler.Accept(token);
            return (int) token;
        }

        void ISampler.Initialize(IQueryParamsContainer query, IEnumerable<int> promptTokens) {
            samplerParams = new RequestSamplingParams(promptTokens);
            if (query.temperature.HasValue) { temperature = query.temperature.Value; }
            if (query.presence_penalty.HasValue) { presence_penalty = query.presence_penalty.Value; }
            if (query.frequency_penalty.HasValue) { frequency_penalty = query.frequency_penalty.Value; }
            if (query.repetition_penalty.HasValue) { repetition_penalty = query.repetition_penalty.Value; }
            if (query.repetition_range.HasValue) { repetition_range = query.repetition_range.Value; }
            if (query.top_p.HasValue) { top_p = query.top_p.Value; }
            if (query.min_p.HasValue) { min_p = query.min_p.Value; }
            if (query.min_tokens.HasValue) { minimum_tokens = query.min_tokens.Value; }
            logit_bias = query.logit_bias?.Select(x => new LLamaLogitBias() { Token = x.Key, Bias = x.Value }).ToArray();

            sampler = CreateChain(Model.instance.context.NativeHandle);
        }

        SafeLLamaSamplerChainHandle CreateChain(SafeLLamaContextHandle context) {
            var chain = SafeLLamaSamplerChainHandle.Create(LLamaSamplerChainParams.Default());

            if (repetition_penalty != 1 || frequency_penalty != 0 || presence_penalty != 0) {
                chain.AddPenalties(
                    context.VocabCount,
                    context.ModelHandle.Tokens.EOS, context.ModelHandle.Tokens.Newline ?? 0,
                    repetition_range, repetition_penalty,
                    frequency_penalty, presence_penalty,
                    penalizeNewline: false, ignoreEOS: true
                );
            }

            chain.AddTopK(top_k);
            chain.AddTailFree(tail_free_z, 1);
            chain.AddTypical(typical_p, 1);
            chain.AddTopP(top_p, 1);
            chain.AddMinP(min_p, 1);
            chain.AddTemperature(temperature);
            if (logit_bias != null) { chain.AddLogitBias(context.VocabCount, logit_bias); }

            chain.AddSoftmax();
            chain.AddDistributionSampler(seed: (uint) Random.Shared.Next(0, int.MaxValue));

            return chain;
        }
    }
}

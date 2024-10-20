using System.Collections.Generic;
using System.Numerics.Tensors;
using System;
using System.Collections.Concurrent;

namespace Llamba.Sampling {
    /// <summary> Sampler that's quick and contains the basic sampling techniques. Customizable and effective for achieving high quality during batching. </summary>
    public class StandardSampler : ISampler {
        public float temperature { get; set; } = 1f;
        public float repetition_penalty { get; set; } = 1f;
        public float presence_penalty { get; set; } = 0;
        public float frequency_penalty { get; set; } = 0;
        public int repetition_range { get; set; } = 2048;
        public int minimum_tokens { get; set; } = 10;

        /// <summary> Toggle this to true if you don't care about finding out what your model can or cannot do. </summary>
        public bool preventRefusals { get; set; } = true;


        public RequestSamplingParams samplerParams { get; set; }
        StandardSamplerRequestParams _samplerParams => samplerParams as StandardSamplerRequestParams;

        int ISampler.SampleToken(Span<float> logits) {
            // Flat-out prevent some tokens from being outputted, if specified.
            if (preventRefusals && samplerParams.receivedTokensCount < minimum_tokens) { SmartBuffer.PreventRefusals(logits); }
            //SmartBuffer.PreventUnwantedWords(logits);

            // Add the logit biases directly to the logits.
            if (_samplerParams.logitBiasBuffer != null) { TensorPrimitives.Add(logits, _samplerParams.logitBiasBuffer, logits); }

            // Penalize repetition by adding the penalty buffers to our logits before sampling.
            if (_samplerParams.frequencyPenaltyBuffer != null) { TensorPrimitives.Add(logits, _samplerParams.frequencyPenaltyBuffer, logits); }
            if (_samplerParams.presencePenaltyBuffer != null) { TensorPrimitives.Add(logits, _samplerParams.presencePenaltyBuffer, logits); }
            if (_samplerParams.distinctTokenList != null) { // For repetition_penalty, divide each existing token by the penalty's value.
                foreach (var token in _samplerParams.distinctTokenList) { logits[token] /= repetition_penalty; }
            }

            // Finally, sample the token with bigger logit after applying some randomization.
            if (temperature != 0) { SmartBuffer.ApplyTemperature(logits, temperature); }
            return TensorPrimitives.IndexOfMax(logits);
        }
        void ISampler.PostSampleInternal(int sampledToken) {
            // Register the new token to the penalty buffers.
            if (_samplerParams.frequencyPenaltyBuffer != null) { _samplerParams.frequencyPenaltyBuffer[sampledToken] -= frequency_penalty; }
            if (_samplerParams.presencePenaltyBuffer != null && _samplerParams.presencePenaltyBuffer[sampledToken] == 0) { _samplerParams.presencePenaltyBuffer[sampledToken] -= presence_penalty; }
            _samplerParams.distinctTokenList?.Add(sampledToken);
        }

        void ISampler.Initialize(IQueryParamsContainer query, IEnumerable<int> promptTokens) {
            samplerParams = new StandardSamplerRequestParams(promptTokens);

            // Cache the desired sampling values passed in the query.
            if (query.temperature.HasValue) { temperature = query.temperature.Value; }
            if (query.presence_penalty.HasValue) { presence_penalty = query.presence_penalty.Value; }
            if (query.frequency_penalty.HasValue) { frequency_penalty = query.frequency_penalty.Value; }
            if (query.repetition_penalty.HasValue) { repetition_penalty = query.repetition_penalty.Value; }
            if (query.repetition_range.HasValue) { repetition_range = query.repetition_range.Value; }
            if (query.min_tokens.HasValue) { minimum_tokens = query.min_tokens.Value; }

            // Pre-calculate the logit bias buffer for reusing during sampling.
            if (query.logit_bias != null) {
                _samplerParams.logitBiasBuffer = SmartBuffer.Rent();
                foreach (var bias_entry in query.logit_bias) { _samplerParams.logitBiasBuffer[bias_entry.Key] = bias_entry.Value; }
            }

            // Pre-create the penalty buffers and calculate their values.
            if (repetition_penalty != 1f || presence_penalty != 0f || frequency_penalty != 0f) {
                if (repetition_penalty != 1f) { _samplerParams.distinctTokenList = StandardSamplerRequestParams.GetOrCreateDistinctTokenList(); }
                if (presence_penalty != 0f) { _samplerParams.presencePenaltyBuffer = SmartBuffer.Rent(); }
                if (frequency_penalty != 0f) { _samplerParams.frequencyPenaltyBuffer = SmartBuffer.Rent(); }

                // Select the last `repetition_range` tokens and apply specified repetition penalty for them.
                var lowerPenalizableTokenIndex = samplerParams.promptTokens.Count - Math.Min(samplerParams.promptTokens.Count, repetition_range);
                for (int i = samplerParams.promptTokens.Count - 1; i > lowerPenalizableTokenIndex; i--) {
                    var sampledToken = samplerParams.promptTokens[i];
                    if (repetition_penalty != 1f) { _samplerParams.distinctTokenList.Add(sampledToken); }
                    if (frequency_penalty != 0f) { _samplerParams.frequencyPenaltyBuffer[sampledToken] -= frequency_penalty; }
                    if (presence_penalty != 0f) { _samplerParams.presencePenaltyBuffer[sampledToken] -= presence_penalty; }
                }
            }
        }

        class StandardSamplerRequestParams : RequestSamplingParams {
            static ConcurrentQueue<HashSet<int>> pool = new();
            public static HashSet<int> GetOrCreateDistinctTokenList() => pool.TryDequeue(out var d) ? d : new(1024);

            public float[] logitBiasBuffer { get; set; }
            public float[] frequencyPenaltyBuffer { get; set; }
            public float[] presencePenaltyBuffer { get; set; }
            public HashSet<int> distinctTokenList { get; set; }

            public StandardSamplerRequestParams(IEnumerable<int> promptTokens) : base(promptTokens) { }

            async public override void Dispose() {
                base.Dispose();
                if (logitBiasBuffer != null) { SmartBuffer.Return(logitBiasBuffer); }
                if (presencePenaltyBuffer != null) { SmartBuffer.Return(presencePenaltyBuffer); }
                if (frequencyPenaltyBuffer != null) { SmartBuffer.Return(frequencyPenaltyBuffer); }
                if (distinctTokenList != null) { pool.Enqueue(distinctTokenList); }
            }
        }
    }
}

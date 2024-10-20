using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Llamba.Sampling {
    /// <summary> Buffer manager responsible for holding available a bunch of buffers-for-rent for logit-related operations. </summary>
    static class SmartBuffer {
        static ConcurrentQueue<float[]> buffers = new(); // Generic buffers big enough to hold the logits.

        static float[] refusalsWeightsBuffer; // Ban specific tokens from appearing first thing in the sequence.
        static float[] newlineWeightsBuffer; // Contains all tokens that introduce new lines -- easy ban!
        static float[] capsWeightsBuffer; // All tokens that contain caps are here to ban if needed.
        static float[] badWeightsBuffer;

        public static void Initialize() {
            refusalsWeightsBuffer = new float[Model.instance.model.VocabCount];
            newlineWeightsBuffer = new float[Model.instance.model.VocabCount];
            capsWeightsBuffer = new float[Model.instance.model.VocabCount];
            badWeightsBuffer = new float[Model.instance.model.VocabCount];

            // TODO: Make these tweakable for end-users. Currently developers can alter them right here.
            var refusals = new[] { "I", " I", " cannot", "cannot", "Cannot", " can't", "can't", "'t", "*I", "*i", "As", "*ahem*", "ahem", "shut down", "clears", " refuse", "Note", "refuses", "<|eot_id|>", "<eos>", "[", " [", "\n\n", "\n", "However" };
            //var unwantedWords = new List<string> { };

            // Initialize easy-ban buffers, so samplers can just add the logit arrays to ban tokens.
            for (int i = 0; i < Model.instance.model.VocabCount; i++) { if (Model.vocab[i].Contains('\n')) { newlineWeightsBuffer[i] = -100; } }
            for (int i = 0; i < Model.instance.model.VocabCount; i++) { if ("ABCDEFGHIJKLMNOPQRSTUVWXYZ".Any(Model.vocab[i].Contains)) { capsWeightsBuffer[i] = -100; } }
            foreach (var token in refusals.SelectMany(Model.instance.Tokenize)) { refusalsWeightsBuffer[token] = -100; } // Add all refusal tokens to the weights buffer,
            foreach (var token in new[] {"*", " *", "\"", " \"", ".", "-"}.Select(x => Model.instance.Tokenize(x)[0])) { refusalsWeightsBuffer[token] = 0; } // ..but not asterisks and standard symbols.
            //foreach (var token in unwantedWords.SelectMany(Model.instance.Tokenize)) { badWeightsBuffer[token] = -100;}
            TemperatureBuffer.Initialize(64);
        }

        /// <summary> Rent a buffer if you wanna store some logit operations during sampling. Don't forget to <b>Return(..)</b> it. </summary>
        public static float[] Rent() => buffers.TryDequeue(out var buffer) ? buffer : new float[Model.instance.model.VocabCount];

        public static void Return(float[] rentedBuffer, bool clear = true) {
            if (clear) { Array.Clear(rentedBuffer); }
            buffers.Enqueue(rentedBuffer);
        }

        public static float Lerp(float from, float to, float t) => from + (to - from) * t;

        public static void PreventRefusals(Span<float> logits) => TensorPrimitives.Add(logits, refusalsWeightsBuffer, logits);
        public static void PreventNewlines(Span<float> logits) => TensorPrimitives.Add(logits, newlineWeightsBuffer, logits);
        public static void PreventCaps(Span<float> logits) => TensorPrimitives.Add(logits, capsWeightsBuffer, logits);
        public static void PreventUnwantedWords(Span<float> logits) => TensorPrimitives.Add(logits, badWeightsBuffer, logits);
        public static Span<float> ApplyTemperature(Span<float> logits, float temperature = 1f) => TemperatureBuffer.ApplyTemperature(logits, temperature);
    }
}
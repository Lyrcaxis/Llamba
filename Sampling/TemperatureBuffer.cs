using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;

namespace Llamba.Sampling {
    /// <summary>
    /// <para> Helper class that holds bunch of buffers responsible for randomizing the logits before sampling. </para>
    /// <para> Internally creates 40 buffers that represent temperature scales of [0, 2], and keeps buffers with unused scales, recomputing them after usage. </para>
    /// <para> Each buffer contains bunch of <b>`float[vocabCount]`</b> sub-buffers that contain randomized numbers according to each buffer's temperature. </para>
    /// <para> Because of how we apply temperature sampling, we're using a transformed scale for temperature, with increments of 0.05f, that alter the min/max multis of the logits. </para>
    /// </summary>
    public class TemperatureBuffer {
        static Dictionary<int, TemperatureBuffer> tempMap = [];
        static HashSet<TemperatureBuffer> allBuffers = [];

        static bool isInitialized = false; // Avoid re-initialization.

        /// <summary> Initializes with 40 buffers representing temperature ranges of [0, 2], and runs a coroutine to keep them up-to-date with randomized scales. </summary>
        public static void Initialize(int initialBufferSize) {
            if (isInitialized) { return; }
            for (int i = 0; i <= 40; i++) { tempMap.Add(i, new TemperatureBuffer(i, initialBufferSize)); } // Temperature range of [0, 2].
            isInitialized = true;

            new Thread(async () => {
                while (true) {
                    bool foundAnyDirty = false;
                    foreach (var tempBuffer in allBuffers) {
                        while (tempBuffer.pool.TryDequeue(out var buffer)) {
                            foundAnyDirty = true;
                            tempBuffer.randoz.Enqueue(tempBuffer.PopulateBuffer(buffer));
                        }
                    }
                    if (!foundAnyDirty) { await Task.Delay(1); }
                }
            }).Start();
        }

        /// <summary> Returns the scaled temperature, representing the amount of 0.05f increments. </summary>
        static int GetTransformedTemperatureScale(float temp) => (int) Math.Round(temp * 20);


        float minT, maxT;

        ConcurrentQueue<float[]> pool = new(); // Buffers that have been used and need re-randomization.
        ConcurrentQueue<float[]> randoz = new(); // Buffers ready to be used for logits randomization.

        TemperatureBuffer(int scaledTemperature, int initialBufferCount) {
            // Scale the temperature by 20 because the scaled one represents number of 0.05f increments.
            var temperature = (float) Math.Round(scaledTemperature * 0.05f, 2);

            // Get the min and max values of the randomization buffers. Inbetween values will act as direct multipliers for the logits when applied.
            minT = 1 - Math.Max(0, (temperature - 1f) / 2f); // Min multi should be 1, unless temperature is > 1.
            maxT = 1 + temperature / 5f; // Max multi of temp/5f. (Rough hand-picked number).
            allBuffers.Add(this);

            for (int i = 0; i < initialBufferCount; i++) { randoz.Enqueue(PopulateBuffer(new float[Model.instance.model.VocabCount])); }
        }

        /// <summary> Chooses the buffer for the specified temperature, then applies the appropriate randomization to the logits. </summary>
        public static Span<float> ApplyTemperature(Span<float> logits, float temperature) {
            var scaledTemperatureIndex = Math.Clamp(GetTransformedTemperatureScale(temperature), 0, tempMap.Count - 1);
            return tempMap[scaledTemperatureIndex].ApplyRando(logits);
        }

        /// <summary> Multiplies the logits with random numbers per token ID, effectively randomizing the logits, causing undeterministic sampling. </summary>
        /// <remarks> The randoz of this instance reflect the specified temperature this buffer represents. </remarks>
        Span<float> ApplyRando(Span<float> logits) {
            if (!randoz.TryDequeue(out var rando)) { rando = PopulateBuffer(new float[Model.instance.model.VocabCount]); }

            TensorPrimitives.Multiply(logits, rando, logits); // Multiply the logits with the random values.
            pool.Enqueue(rando); // Return the randomization buffer to the pool, so it'll be recomputed.
            return logits;
        }

        float[] PopulateBuffer(float[] buffer) {
            float Lerp(float from, float to, float t) => from + (to - from) * t;

            // Apply random multiplier from `minT` to `maxT` for each cell. This'll be used to randomize the logits.
            for (int i = 0; i < buffer.Length; i++) { buffer[i] = Lerp(minT, maxT, Random.Shared.NextSingle()); }
            buffer[Model.instance.eotID] = Math.Min(buffer[Model.instance.eotID], 1); // Do not scale eotID. Found this to work best.
            return buffer;
        }

    }
}

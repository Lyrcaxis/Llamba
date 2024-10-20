using LLama.Native;

using Llamba.Server;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Llamba.Batching {
    /// <summary> Processor of the batches, decoder of the tokens, and core component of the inference loop. </summary>
    /// <remarks> Responsible for decoding as many requests as possible from the 'queued' requests, as quick as possible. </remarks>
    public class BatchProcessor {
        public readonly ConcurrentQueue<InferenceRequest> queued = []; // Any requests that should be processed should go here.
        readonly List<InferenceRequest> active = new(1000); // Requests that are being processed. Internal and automatically populated by queued.

        readonly SafeLLamaContextHandle context;
        readonly int vocabCount;
        readonly int maxContextSize;
        readonly int maxBatchSize;
        readonly LLamaBatch batch = new();
        readonly bool debug;

        public BatchProcessor(SafeLLamaContextHandle context, bool debug = false) {
            (this.context, this.debug) = (context, debug);
            maxBatchSize = (int) context.BatchSize;
            maxContextSize = (int) context.ContextSize;
            vocabCount = context.ModelHandle.VocabCount;

            new Thread(MainLoop).Start(); // Initialize the main loop checks
        }

        /// <summary> The main loop: Bring 'queued' requests to 'active', and perform inference to get the next tokens, in a single batch. </summary>
        async void MainLoop() {
            var (i, stack, a) = (0, 0, DateTime.Now);
            while (true) { // Refresh the context, then infer for next logits.
                if (active.Count == 0 && queued.IsEmpty) { await Task.Delay(1); continue; }
                ContextRefresher.FullContextRefresh(context, batch, maxContextSize, maxBatchSize, active, queued);
                if (active.Count == 0) { await Task.Delay(1); continue; }
                InferNextTokenLogits(); // Run the next inference step with the latest tokens in a single batch, so we can get the next tokens.
                if (debug) { Debug(); } // Log debug stats if needed. Note the time printed will also include prompt processing for new requests.
            }

            void Debug() {
                stack += active.Count;
                if (i % 10 != 0) { i++; return; }
                var b = DateTime.Now;
                if (++i != 0) { Console.WriteLine($"{stack:d4} tokens in {(b - a).TotalSeconds:f2}s ({stack / (b - a).TotalSeconds:f2} T/s) -- {active.Count:d3} active, {queued.Count:d4} in queue."); }
                (stack, a) = (0, DateTime.Now);
            }
        }

        /// <summary> Performs inference, sending only the latest tokens in each sequence for decoding, and retrieving their logits. </summary>
        /// <remarks> Once the logits are retrieved, they're sent to the corresponding request for sampling or whatever. </remarks>
        void InferNextTokenLogits() {
            if (batch.TokenCount + active.Count > maxBatchSize) { DecodeAndClear(); } // Make sure the active requests fit in one batch.
            foreach (var r in active) { batch.Add(r.sampledToken, pos: r.totalTokens, sequence: (LLamaSeqId) r.sequenceID, logits: true); }
            DecodeAndClear(); // Decode only the last tokens in the sequence -- these are the ones that'll require logits.
            var newLogits = GetLogits(active.Count); // Inference with the latest tokens in batch, and sent logits for sampling.
            for (int i = 0; i < active.Count; i++) { active[i].HandleLogits(newLogits.Slice(i * vocabCount, vocabCount)); }
        }

        void DecodeAndClear() { context.Decode(batch); batch.Clear(); }
        unsafe Span<float> GetLogits(int seqAmount) => new(llama_get_logits(context), seqAmount * vocabCount); // C# port of c++'s `llama_get_logits()`
        [DllImport("llama", CallingConvention = CallingConvention.Cdecl)] public unsafe static extern float* llama_get_logits(SafeLLamaContextHandle ctx);
    }
}
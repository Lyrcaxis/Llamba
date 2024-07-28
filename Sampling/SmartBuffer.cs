using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;

namespace Llamba.Sampling {
	/// <summary> Buffer manager responsible for holding available a bunch of buffers-for-rent for logit-related operations. </summary>
	static class SmartBuffer {
		const int bufferCount = 1024; // TODO: Be more generous if we have the RAM and we want to handle more concurrent requests.
		const float minT = 0.9f, maxT = 1.1f; // Logit randomization range. TODO: Make it temperature-based.

		static int vocabCount; // Taken straight from the model's vocab count.
		static ConcurrentQueue<float[]> buffers = new(); // Generic buffers big enough to hold the logits.
		static ConcurrentQueue<float[]> randoz  = new(); // Buffers specific to the randomization of logits.

		static float[] refusalsWeightsBuffer; // Ban specific tokens from appearing first thing in the sequence.
		static float[] repetitionWeightsBuffer; //TODO: Weight repetition differently for different tokens.
		static float[] newlineWeightsBuffer; // Contains all tokens that introduce new lines -- easy ban!
		static float[] foreverBannedWeights; // This should map -100s to every token we don't want to ever see.
		static float[] capsContainedWeights; // All tokens that contain caps are here to ban if needed.

		static SmartBuffer() {
			vocabCount = Model.instance.model.VocabCount;

			new Thread(async () => {
				refusalsWeightsBuffer = new float[vocabCount];
				foreverBannedWeights = new float[vocabCount];
				newlineWeightsBuffer = new float[vocabCount];
				capsContainedWeights = new float[vocabCount];
				repetitionWeightsBuffer = Model.vocab.Select(x => 1f).ToArray(); // TODO: Make this tweakable for end-users.

				// TODO: Make these tweakable for end-users. Currently developers can alter them right here.
				var bannedRefusals = new[] { "I", " I", " cannot", "cannot", "Cannot", " can't", "can't", "*I", "*i", "As", "*ahem*", "ahem", "shut down", "clears", " refuse", "refuse", "refuses", "<|eot_id|>", "\n\n", "\n" };
				var bannedTokens = new List<string> { "No", "N", "*I", "no", " no", " not", "Not", " Not", "W", "\"W", "I", " don't", "Newsflash", " Newsflash", " newsflash", " buddy", "Please", " Please", " widen", "Oh", "'s" };

				// Initialize easy-ban buffers, so samplers can just add the logit arrays to ban tokens.
				for (int i = 0; i < vocabCount; i++) { if (Model.vocab[i].Contains('\n')) { newlineWeightsBuffer[i] = -100; } }
				for (int i = 0; i < vocabCount; i++) { if ("ABCDEFGHIJKLMNOPQRSTUVWXYZ".Any(Model.vocab[i].Contains)) { capsContainedWeights[i] = -100; } }
				foreach (var token in bannedRefusals.SelectMany(x => Model.instance.Tokenize(x))) { refusalsWeightsBuffer[token] = -100; } // Ban refusal tokens
				foreach (var token in new[] {"*", " *"}.Select(x => Model.instance.Tokenize(x)[0])) { refusalsWeightsBuffer[token] = 0; } // .. but not asterisks..
				foreach (var token in bannedTokens) { foreverBannedWeights[Model.instance.Tokenize(token)[0]] = -100; } // Ban tokens that should always be banned.

				// Then, initialize the buffer pool, so requests can rent from here instead of creating them.
				for (int i = 0; i < bufferCount; i++) { buffers.Enqueue(new float[vocabCount]); }
				for (int i = 0; i < bufferCount; i++) { randoz.Enqueue(Enumerable.Range(0, vocabCount).Select(x => Lerp(minT, maxT, Random.Shared.NextSingle())).ToArray()); }

				// And finally initialize the coroutine for keeping the randomization buffers up asynchronously. QuickSampler requires one randomization buffer per output token.
				while (true) {
					if (randoz.Count >= bufferCount) { await Task.Delay(1); continue; }

					var bufferToMakeRando = Rent();
					for (int i = 0; i < bufferToMakeRando.Length; i++) { bufferToMakeRando[i] = Lerp(minT, maxT, Random.Shared.NextSingle()); }
					randoz.Enqueue(bufferToMakeRando);
				}
			}).Start();
		}

		/// <summary> Rent a buffer if you wanna store some logit operations during sampling. Don't forget to <b>Return(..)</b> it. </summary>
		public static float[] Rent() => buffers.TryDequeue(out var buffer) ? buffer : new float[vocabCount];

		/// <summary> Rent a 'randomization buffer' if you wanna sample randomly from some logits. Don't forget to <b>Return(..)</b> it. </summary>
		static float[] RentRando() => randoz.TryDequeue(out var rando) ? rando : Enumerable.Range(0, vocabCount).Select(x => Lerp(minT, maxT, Random.Shared.NextSingle())).ToArray();
		public static void Return(float[] rentedBuffer, bool clear = true) {
			if (clear) { Array.Clear(rentedBuffer); }
			buffers.Enqueue(rentedBuffer);
		}

		static float Lerp(float from, float to, float t) => from + (to - from) * t;

		public static void PreventRefusals(Span<float> logits) => TensorPrimitives.Add(logits, refusalsWeightsBuffer, logits);
		public static void PreventNewlines(Span<float> logits) => TensorPrimitives.Add(logits, newlineWeightsBuffer, logits);
		public static void PreventBadTokens(Span<float> logits) => TensorPrimitives.Add(logits, foreverBannedWeights, logits);
		public static void PreventCaps(Span<float> logits) => TensorPrimitives.Add(logits, capsContainedWeights, logits);
		public static Span<float> ApplyRando(Span<float> logits) { var rando = RentRando(); TensorPrimitives.Multiply(logits, rando, logits); Return(rando); return logits; }
	}
}
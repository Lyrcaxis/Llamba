using LLama;
using LLama.Common;

using Llamba.Batching;
using Llamba.Sampling;
using Llamba.Tokenization;
using Llamba.Server;

using System;
using System.Collections.Generic;

namespace Llamba {
	public class Model {
		public LLamaWeights model { get; init; }
		public LLamaContext context { get; init; }
		public static Dictionary<int, string> vocab { get; private set; }

		public Func<ISampler> SamplerFactoryFunc { get; set; } = () => new StandardSampler();

		public int eotID { get; private set; }

		BatchProcessor processor;
		ModelParams modelParams;
		IInferenceFormat format;

		public static Model instance { get; private set; }
		public Model(string modelPath, IInferenceFormat format) {
			(instance = this).format = format;

			modelParams = new ModelParams(modelPath) {
				ContextSize = 32 * 1024,
				GpuLayerCount = 99,
				BatchSize = 32 * 1024,
				UseMemoryLock = true,
				UseMemorymap = true,
				//TypeK = LLama.Native.GGMLType.GGML_TYPE_F32,
				//TypeV = LLama.Native.GGMLType.GGML_TYPE_F32,
				//FlashAttention = true
			};
			model = LLamaWeights.LoadFromFile(modelParams);

			vocab = model.GetVocab(); vocab[-1] = "";
			if (format is LLama3Format) { vocab[128009] = "<|eot_id|>"; }
			eotID = Tokenize(format.EOT)[0];
			context = model.CreateContext(modelParams);
			SmartBuffer.Return(SmartBuffer.Rent()); // Get the SmartBuffer going
			processor = new(context.NativeHandle, debug: false); // Initialize the processor
		}

		public InferenceRequest AddRequest(ChatQuery query) {
			var prompt = format.TurnToString(query.messages, !(query.@continue ?? false)) + ((query.@continue ?? false) ? "" : query.appendText);
			var request = new InferenceRequest(prompt, query, Tokenize(prompt), SamplerFactoryFunc());
			processor.queued.Enqueue(request);
			return request;
		}
		public void RemoveRequest(InferenceRequest request) => request.needsGen = false;

		public string Format(ChatMessage[] messages, bool includeGenerationPrompt = true) => format.TurnToString(messages, includeGenerationPrompt);
		public List<int> Tokenize(string text) => Tokenizer.Encode(text);
		public List<int> Tokenize(ChatMessage[] messages) => Tokenize(Format(messages));
	}
}

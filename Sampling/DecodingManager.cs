using Llamba.Server;

using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace Llamba.Sampling {
	/// <summary> Module that acts as a pool for decoding non-character tokens. Needed for emojis and non-english characters. </summary>
	/// <remarks> The decoders are ONLY used if non-character tokens are met (e.g.: some emojis are consisted of 3+ tokens). </remarks>
	public static class DecodingManager {
		static Dictionary<InferenceRequest, InferenceDecoder> decoderMap = [];
		static Queue<InferenceDecoder> decoderPool = [];

		/// <summary> Attempts to decode a sequence of non-character tokens. Internally adds to the per-request decoder so no text will be missed. </summary>
		/// <remarks> This will effectively ONLY return output if the sequence of non-character tokens complete a unicode character. </remarks>
		public static bool Decode(InferenceRequest request, out string output, int sampledToken) {
			if (!decoderMap.TryGetValue(request, out var decoder)) { decoderMap.Add(request, decoderPool.TryDequeue(out decoder) ? decoder : decoder = new()); }
			return decoder.Decode(out output, sampledToken);
		}

		/// <summary> Disposes the decoder for a specific request and adds it to the pool of cached decoders for future use. </summary>
		public static void ReturnDecoder(InferenceRequest request) {
			if (!decoderMap.TryGetValue(request, out var decoder)) { return; }
			decoder.decoder.Reset();
			decoderPool.Enqueue(decoder);
			decoderMap.Remove(request);
		}

		class InferenceDecoder {
			public Decoder decoder = Encoding.UTF8.GetDecoder();

			public bool Decode(out string output, int sampledToken) {
				var charBuffer = ArrayPool<char>.Shared.Rent(32);
				var byteBuffer = ArrayPool<byte>.Shared.Rent(32);
				try {
					var bytesCount = Model.instance.model.NativeHandle.TokenToSpan(sampledToken, byteBuffer);
					decoder.Convert(byteBuffer, 0, (int) bytesCount, charBuffer, 0, charBuffer.Length, false, out var _, out var charsUsed, out bool _);
					output = new string(charBuffer, 0, charsUsed);
					return !output.StartsWith('�');
				}
				finally {
					ArrayPool<char>.Shared.Return(charBuffer);
					ArrayPool<byte>.Shared.Return(byteBuffer);
				}
			}
		}

	}
}

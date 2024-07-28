using System;
using System.Collections.Generic;
using System.Text;

namespace Llamba {
	public interface IInferenceFormat {
		/// <summary> The begin-of-sequence text. </summary>
		string BOS { get; }

		/// <summary> The end-of-turn text. </summary>
		string EOT { get; }

		/// <summary> Function that converts a message/turn into a string. </summary>
		Func<ChatMessage, string> formatTransformFunc { get; }

		/// <summary> Converts a collection of messages into a single string. </summary>
		/// <remarks> Can optionally append the generation prompt marking the beginning of a new turn. </remarks>
		public string TurnToString(IList<ChatMessage> messages, bool includeGenerationPrompt = true) {
			var sb = new StringBuilder(BOS);
			foreach (var msg in messages) { sb.Append(formatTransformFunc(msg)); }
			if (includeGenerationPrompt) { sb.Append(formatTransformFunc(new("", Role.Assistant))); }

			// Finally, return it without the last EOS, and with LF (Unix) line endings.
			return sb.ToString(0, sb.Length - EOT.Length).Replace("\r\n", "\n").TrimStart();
		}
	}

	public class LLama3Format : IInferenceFormat {
		public string BOS { get; init; } = "<|begin_of_text|>";
		public string EOT { get; init; } = "<|eot_id|>";
		public Func<ChatMessage, string> formatTransformFunc => (msg) => $"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}{EOT}";

		public LLama3Format() { }
		public LLama3Format(string EOT) => this.EOT = EOT;
		public LLama3Format(string BOS, string EOT) => (this.BOS, this.EOT) = (BOS, EOT);
	}
	public class ChatMLFormat : IInferenceFormat {
		public string BOS { get; init; } = "";
		public string EOT { get; init; } = "<|im_end|>";
		public Func<ChatMessage, string> formatTransformFunc => (msg) => $"\n<|im_start|>{msg.role}\n{msg.content}{EOT}";

		public ChatMLFormat() { }
		public ChatMLFormat(string EOT) => this.EOT = EOT;
		public ChatMLFormat(string BOS, string EOT) => (this.BOS, this.EOT) = (BOS, EOT);
	}
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
            //for (int i = 0; i < 10; i++) { Debug.WriteLine(""); } Debug.WriteLine(sb.ToString(0, sb.Length - EOT.Length).Replace("\r\n", "\n").TrimStart());

            // Finally, return it without the last EOS, and with LF (Unix) line endings.
            return sb.ToString(0, sb.Length - EOT.Length).Replace("\r\n", "\n").TrimStart();
        }
    }

    public class LLama3Format : IInferenceFormat {
        public string BOS { get; init; } = "<|begin_of_text|>";
        public string EOT { get; init; } = "<|eot_id|>";
        public Func<ChatMessage, string> formatTransformFunc => (msg) => $"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}{EOT}";
    }
    public class ChatMLFormat : IInferenceFormat {
        public string BOS { get; init; } = "";
        public string EOT { get; init; } = "<|im_end|>";
        public Func<ChatMessage, string> formatTransformFunc => (msg) => $"\n<|im_start|>{msg.role}\n{msg.content}{EOT}";
    }
    public class MistralFormat : IInferenceFormat {
        public string BOS { get; init; } = "<s>";
        public string EOT { get; init; } = "</s>";
        public Func<ChatMessage, string> formatTransformFunc => (msg) =>
            msg.role switch {
                "assistant" => $"[/INST] {msg.content}</s>",
                _ => $"[INST] {msg.content}",
            }; // This would actually be alright except the final space would be appended in the prompt.

        /// <summary> Custom format transform for Mistral models, since they do not support system messages. </summary>
        public string TurnToString(IList<ChatMessage> messages, bool includeGenerationPrompt = true) {
            var sb = new StringBuilder(BOS);

            foreach (var m in messages) {
                sb.Append(m.role switch {
                    "assistant" => $"[/INST] {m.content.Trim()}</s>",
                    _ => $"[INST] {m.content.Trim()}"
                });
            }

            if (includeGenerationPrompt) { sb.Append($"[/INST]"); }
            else if (sb.ToString().EndsWith(EOT)) { sb.Length -= EOT.Length; }
            return sb.Replace("\r\n", "\n").Replace("\n ", "\n").Replace(" \n", "\n").ToString().TrimStart();
        }
    }

    public class Gemma2Format : IInferenceFormat {
        public string BOS { get; init; } = "<bos>";
        public string BOT { get; init; } = "<start_of_turn>";
        public string EOT { get; init; } = "<end_of_turn>";
        public Func<ChatMessage, string> formatTransformFunc => throw new NotImplementedException();

        /// <summary> Custom format transform for Gemma 2 models, since they do not support system messages. </summary>
        public string TurnToString(IList<ChatMessage> messages, bool includeGenerationPrompt = true) {
            var sb = new StringBuilder(BOS);

            foreach (var m in messages) {
                var (role, content, isSystem) = (m.role, m.content.Trim(), m.role == "system");
                if (isSystem) { (role, content) = ("user", $"[SYSTEM:\n{content}\n]"); }
                sb.Append($"\n{BOT}{role}\n{content}{EOT}");
                if (isSystem) { sb.Append($"\n{BOT}assistant\n[Aknowledged.]{EOT}"); }
            }

            if (includeGenerationPrompt) { sb.Append($"\n{BOT}assistant\n"); }
            else if (sb.ToString().EndsWith(EOT)) { sb.Length -= EOT.Length; }

            sb = sb.Replace(">assistant", ">model").Replace(">next", ">system").Replace("<bos>\n", "<bos>");
            return sb.Replace("\r\n", "\n").Replace("\n ", "\n").Replace(" \n", "\n").ToString().TrimStart();
        }
    }

}

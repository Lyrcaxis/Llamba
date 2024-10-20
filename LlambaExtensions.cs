using LLama;
using LLama.Native;

using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Llamba {
    public static class LlambaExtensions {
        static Decoder decoder = Encoding.UTF8.GetDecoder();
        public static Dictionary<int, string> GetVocab(this LLamaWeights model) {
            var (bytesArr, charsArr) = (new byte[128], new char[128]);
            return Enumerable.Range(0, model.VocabCount).ToDictionary(i => i, i => {
                decoder.Convert(bytesArr, 0, (int) model.NativeHandle.TokenToSpan(i, bytesArr), charsArr, 0, charsArr.Length, true, out var _, out var charsUsed, out var _);
                return string.Join("", charsArr.Take(charsUsed));
            });
        }

        public static void ClearSeq(this SafeLLamaContextHandle context, int seq) => NativeApi.llama_kv_cache_seq_rm(context, seq: (LLamaSeqId) seq, p0: 0, p1: -1);
        public static void RemoveSeq(this SafeLLamaContextHandle context, int seq, int from, int until) => NativeApi.llama_kv_cache_seq_rm(context, seq: (LLamaSeqId) seq, p0: from, p1: until);
        public static void MoveSeq(this SafeLLamaContextHandle context, int seqSrc, int seqDst) {
            NativeApi.llama_kv_cache_seq_cp(context, src: (LLamaSeqId) seqSrc, dest: (LLamaSeqId) seqDst, -1, -1);
            NativeApi.llama_kv_cache_seq_rm(context, seq: (LLamaSeqId) seqSrc, p0: 0, p1: -1);
        }
    }
}

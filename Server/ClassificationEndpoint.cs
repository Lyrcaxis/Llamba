using System.Collections.Generic;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Text;
using System.Linq;

namespace Llamba.Server {
    public class ClassificationEndpoint {
        //static ConcurrentDictionary<string, int> cachedTokenMap = [];
        //static Dictionary<string, Dictionary<int, float>> cachedBiasesMap = [];

        public ClassificationEndpoint(WebApplication app) {
            app.MapPost("/classify", async (context) => await Classify(await JsonSerializer.DeserializeAsync<ClassificationQuery>(context.Request.Body), context));
        }

        async Task Classify(ClassificationQuery query, HttpContext context) {
            try {
                string qResponse = $"As a classifier, I understand your query to answer following question:\n{query.question}.\n\n---\n\nHere are the options you provided:\n{ListToString(query.validResponses, true)}\n---\n\n>...Examining Context...\n>...Complete!\n\nOut of the options you provided, the most reasonable one to answer this question is option #";
                var qQuery = new ChatQuery() { temperature = 0.0001f, repetition_range = 0, repetition_penalty = 0, max_tokens = 1, stream = false, @continue = true };
                qQuery.logit_bias = GetBiases(query.validResponses); // Bias to only allow the specific tokens to be outputted. This makes the model respond with one of the available options.
                qQuery.messages = [new(query.prompt, Role.System), new($"Respond to the question based on the above context. Choose from one of the provided options.", Role.User), new(qResponse, Role.Assistant)];


                await using var sw = new StreamWriter(context.Response.Body);
                using var request = Model.instance.AddRequest(qQuery);
                while (true) {
                    if (request.nextResponse.TryDequeue(out var r)) {
                        await sw.WriteLineAsync(r.response);
                        await sw.FlushAsync();
                        break;
                    }
                    await Task.Delay(1);
                }
            }
            catch (Exception e) { Debug.WriteLine($"{e}\n{e.Message}"); }
        }

        List<string> classes = Enumerable.Range(0, 20).Select(x => x.ToString()).ToList();
        List<int> classesTokenMap = new();
        void InitFakeClassesMap() { for (int i = 0; i < classes.Count; i++) { classesTokenMap.Add(Model.instance.Tokenize(classes[i])[0]); } }
        //List<string> classes = [" first", " second", " third", " fourth", " fifth", " sixth", " seventh", " eighth", " ninth", " tenth", " eleventh", " twelfth", " fourteenth", " fifteenth", " sixteenth", " seventeenth", " eighteenth", " nineteenth", " twentieth"];


        /// <summary> Forms the logit bias, allowing only the tokens responding to the valid responses to be selected. </summary>
        /// <remarks> (NOTE) Right now only allows the LLM to output numbers [1-N], based on the amount of options. </remarks>
        Dictionary<int, float> GetBiases(List<string> validResponses) {
            if (classesTokenMap.Count == 0) { InitFakeClassesMap(); }
            var d = new Dictionary<int, float>();
            for (int i = 0; i < validResponses.Count; i++) { d.Add(classesTokenMap[i], 100); }
            return d;
        }

        static string ListToString(List<string> validResponse, bool numbered = false) {
            var sb = new StringBuilder();
            for (int i = 0; i < validResponse.Count; i++) {
                if (numbered) { sb.AppendLine($"{i + 1}. {validResponse[i]}"); }
                else { sb.AppendLine(validResponse[i]); }
            }
            return sb.Replace("\r\n", "\n").ToString();
        }
    }


    struct ClassificationQuery {
        public string prompt { get; set; }
        public string question { get; set; }
        public List<string> validResponses { get; set; }
    }
}
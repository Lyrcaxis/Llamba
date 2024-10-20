using Llamba.Sampling;
using Llamba.Server;

using Microsoft.AspNetCore.Builder;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llamba.Tests {
    public class BatchingTest {
        public class ConvoExport { public string id { get; set; } public ChatMessage[] messages { get; set; } }
        public class ConvoImport { public ConvoBody body { get; set; } public class ConvoBody { public ChatMessage[] messages { get; set; } } }

        int totalBatches;
        int tokensToRequest;

        Dictionary<string, string> testResults = new();
        Dictionary<string, (string link, string tokensPerSec, int totalTokens)> displayResults = new();

        public BatchingTest(WebApplication app) {
            app.MapGet("/tests/batching", async (context) => {
                await using var sw = new StreamWriter(context.Response.Body);
                await sw.WriteLineAsync(testResults[context.Request.Query["ID"]]);
                await sw.FlushAsync();
            });

            _ = DelayedInit();
        }
        async Task DelayedInit() {
            while (!TemperatureBuffer.isInitialized) { await Task.Delay(1000); }

            (totalBatches, tokensToRequest) = (100, 100);

            var sharedPrompt = "You are Jonas, knower of everything and master of nothing. You will take a role in this adventure and guide the player towards a creative and motivating journey!";
            await TestAllSamplers([new($"{sharedPrompt} Super short response.", Role.System)], "mini-responses");
            await TestAllSamplers([new($"{sharedPrompt} Short response.", Role.System)], "smol-responses");
            await TestAllSamplers([new($"{sharedPrompt} Long response.", Role.System)], "long-responses");

            AnnounceResults();
        }

        async Task TestAllSamplers(ChatMessage[] messages, string key) {
            //Model.instance.SamplerFactoryFunc = () => new QuickSampler() { preventRefusals = false, singleLine = false, penalizeRepetition = false }; // The quickest version.
            //await CreateAndAwaitRequests(messages, $"{key}-QuickSampler");

            Model.instance.SamplerFactoryFunc = () => new StandardSampler() { temperature = 1.7f, repetition_penalty = 1.1f }; // The quick and extensible version.
            await CreateAndAwaitRequests(messages, $"{key}-StandardSampler");

            Model.instance.SamplerFactoryFunc = () => new LLamaSampler() { temperature = 0.8f, repetition_penalty = 1.1f, top_p = 0.9f, min_p = 0.1f }; // The somewhat slower llama sampler version.
            await CreateAndAwaitRequests(messages, $"{key}-LLamaSampler");
        }

        async Task CreateAndAwaitRequests(ChatMessage[] messages, string testID) {
            Console.ForegroundColor = ConsoleColor.Magenta; Console.WriteLine($"\nStarting test for '{testID}'."); Console.ForegroundColor = ConsoleColor.Gray;

            ChatQuery query = new() { messages = messages, max_tokens = tokensToRequest };

            var (beginTime, totalTokens) = (DateTime.Now, 0);
            var requests = Enumerable.Range(0,totalBatches).Select(x => Model.instance.AddRequest(query)).ToList();
            var queryCompletionMap = requests.Select(x => false).ToList();
            var wholeResponses = requests.Select(x => "").ToList();

            while (queryCompletionMap.Any(isComplete => !isComplete)) {
                for (int i = 0; i < requests.Count; i++) {
                    if (queryCompletionMap[i] || GrabResponse(i) is not LocalResponse response) { continue; }
                    wholeResponses[i] += response.delta;
                }
                await Task.Delay(1);
            }

            Finalize();

            LocalResponse GrabResponse(int id) {
                var request = requests[id];
                if (!request.nextResponse.TryDequeue(out var r)) {
                    if (!request.needsGen) {
                        (request as IDisposable).Dispose();
                        queryCompletionMap[id] = true;
                    }
                    return null;
                }
                if (r.response != "") { totalTokens++; }
                return new LocalResponse(default, r.response, r.stopReason);
            }


            void Finalize() {
                var dashes = "\n--------------------------------------------------------\n";
                var sb = new StringBuilder();
                for (int i = 0; i < requests.Count; i++) { sb.AppendLine($"{dashes}{i + 1}:\n{wholeResponses[i]}{dashes}"); }
                testResults.Add(testID, sb.ToString());

                var totalTime = (DateTime.Now - beginTime).TotalSeconds;
                Console.ForegroundColor = ConsoleColor.Green; Console.WriteLine($"Completed {totalBatches} requests in {totalTime:f2}s, for a total of {totalTokens} tokens. ({totalTokens / totalTime:f2}T/s)");
                Console.ForegroundColor = ConsoleColor.White; Console.Write("(You can see the results at ");
                Console.ForegroundColor = ConsoleColor.Blue; Console.Write(Link(testID, $"http://localhost:5059/tests/batching?ID={testID}"));
                Console.ForegroundColor = ConsoleColor.White; Console.WriteLine(")\n");
                Console.ForegroundColor = ConsoleColor.Gray;

                displayResults[testID] = ($"http://localhost:5059/tests/batching?ID={testID}", $"{totalTokens / totalTime:f2}T/s", totalTokens);
            }
        }

        static string Link(string header, string link) => $"\x1B]8;;{link}\x1B\\{header}\x1B]8;;\x1B\\";
        static void PrintSpaces(int spacesAmount) { for (int i = 0; i < spacesAmount; i++) { Console.Write(" "); } }
        void AnnounceResults() {
            var maxLK = displayResults.Max(x => x.Key.Length);
            var maxLT = displayResults.Max(x => x.Value.totalTokens.ToString().Length);
            Console.WriteLine("\n------------------- Test completed -------------------\n");

            // Write the ID Header
            Console.ForegroundColor = ConsoleColor.Magenta;
            PrintSpaces(1); Console.Write("Web Link:");
            PrintSpaces(maxLK - 5); Console.Write("Tokens:");
            PrintSpaces(5); Console.WriteLine("T/S:");


            foreach (var (id, (link, tokensPerSec, totalTokens)) in displayResults) {
                var ws1 = maxLK - id.Length + 5; // Spaces after key
                var ws2 = maxLT - totalTokens.ToString().Length + 4; // Spaces after Tokens

                Console.ForegroundColor = ConsoleColor.Yellow; Console.Write("- ");
                Console.ForegroundColor = ConsoleColor.White; Console.Write($"{Link(id, link)}");
                for (int i = 0; i < ws1; i++) { Console.Write(" "); } Console.ForegroundColor = ConsoleColor.Yellow; Console.Write(totalTokens);
                for (int i = 0; i < ws2; i++) { Console.Write(" "); } Console.ForegroundColor = ConsoleColor.Green; Console.WriteLine(tokensPerSec);
            }
            Console.ForegroundColor = ConsoleColor.Gray;
        }
    }
}

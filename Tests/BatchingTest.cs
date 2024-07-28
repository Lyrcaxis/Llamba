using Llamba.Sampling;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace Llamba.Tests {
	public class BatchingTest {
		public class ConvoExport { public string id { get; set; } public ChatMessage[] messages { get; set; } }
		public class ConvoImport { public ConvoBody body { get; set; } public class ConvoBody { public ChatMessage[] messages { get; set; } } }

		int totalBatches;
		int tokensToRequest;
		int remainingRequests;
		int totalReceivedTokens;

		DateTime beginTime;

		ConcurrentDictionary<int, string> testResponses = new();

		public BatchingTest() {
			Server.Server.app.MapGet("/tests/batching", SeeResults);

			remainingRequests = totalBatches = 1500;
			tokensToRequest = 10;

			Model.instance.SamplerFactoryFunc = () => new QuickSampler() { preventRefusals = false, singleLine = false, penalizeRepetition = false }; // The quickest version
			//Model.instance.SamplerFactoryFunc = () => new QuickSampler() { preventRefusals = true, singleLine = true, penalizeRepetition = true }; // The not-as-quick version (customizable tho)
			//Model.instance.SamplerFactoryFunc = () => new LLamaSampler() { temperature = 0.5f, repetition_penalty = 1.1f, top_p = 0.9f }; // The terribly slow llama sampler version.

			beginTime = DateTime.Now;
			new System.Threading.Thread(() => {
				var messages = new ChatMessage[] { new("You are Jonas, knower of everything and master of nothing. You will take a role in this adventure and guide the player towards a creative and motivating journey! Calm down, though.", Role.System) };
				for (int ID = 0; ID < totalBatches; ID++) { _ = ChatRequest(messages, tokensToRequest, ID); }

				// This can substitute the test above, works with OpenAI-compatible jsonl files.
				//var prompts = new string[totalBatches];
				//var jsonlPath = "";
				//var load = System.IO.File.ReadAllLines(jsonlPath).Take(totalBatches).Select(x => System.Text.Json.JsonSerializer.Deserialize<ConvoImport>(x).body.messages).ToArray();
				//for (int ID = 0; ID < totalBatches; ID++) { _ = ChatRequest(load[ID], tokensToRequest, ID); }

			}).Start();
		}

		async Task ChatRequest(ChatMessage[] messages, int tokens, int ID) {
			int receivedTokens = 0;
			using var request = Model.instance.AddRequest(new() { messages = messages, max_tokens = tokens });
			var sb = new StringBuilder();
			while (request.needsGen == true) { GrabResponses(); await Task.Delay(1); }
			GrabResponses(); // One final time to make sure nothing sneaked in async.
			testResponses.TryAdd(ID, sb.ToString());

			lock (this) {
				totalReceivedTokens += receivedTokens;
				if (--remainingRequests <= 0) {
					var totalTime = (DateTime.Now - beginTime).TotalSeconds;
					Console.ForegroundColor = ConsoleColor.Green;
					Console.WriteLine($"Completed {totalBatches} requests in {totalTime:f2}s, for a total of {totalReceivedTokens} tokens. ({totalReceivedTokens / totalTime:f2}T/s)");
					Console.ForegroundColor = ConsoleColor.White;
					Console.Write("You can see the results at ");
					Console.ForegroundColor = ConsoleColor.Blue;
					Console.WriteLine("http://localhost:5059/tests/batching");
					Console.ForegroundColor = ConsoleColor.Gray;
				}
			}

			void GrabResponses() { while (request.nextResponse.TryDequeue(out var r)) { sb.Append(r.response); receivedTokens += r.tokensCount; } }
		}

		async Task SeeResults(HttpContext context) {
			await using var sw = new StreamWriter(context.Response.Body);
			foreach (var (key, val) in testResponses) { await sw.WriteLineAsync($"{key}: {val}"); }
			await sw.FlushAsync();
		}
	}
}

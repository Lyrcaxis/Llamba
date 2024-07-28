using System.Collections.Generic;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace Llamba.Server {
	public class ChatEndpoint {
		JsonSerializerOptions options = new();
		Model model;

		public ChatEndpoint(WebApplication app, Model model) {
			app.MapPost("/chat/completion", async (context) => await Chat(await JsonSerializer.DeserializeAsync<ChatQuery>(context.Request.Body), context));
			app.MapPost("/chat/tokenize", async (context) => await Tokenize(await JsonSerializer.DeserializeAsync<ChatQuery>(context.Request.Body), context));
			this.model = model;
		}

		async Task Chat(ChatQuery query, HttpContext context) {
			for (int i = 1; i < query.messages.Length; i++) { if (query.messages[i].role == "system") { query.messages[i].role = "next"; } }
			try {
				await using var sw = new StreamWriter(context.Response.Body);
				using var request = model.AddRequest(query);
				while (request.needsGen == true) { await GrabResponses(); await Task.Delay(1); }
				await GrabResponses(); // One final time to make sure nothing sneaked in async.
				if (query.stream != true) { await sw.FlushAsync(); } // Flush if not stream.

				async Task GrabResponses() {
					while (request.nextResponse.TryDequeue(out var r)) {
						if (query.stream != true) { await sw.WriteAsync(r.response); continue; }
						// If we're streaming, write a json response to the body, 
						var response = new LocalResponse(default, r.response, r.stopReason);
						await sw.WriteLineAsync($"data: {JsonSerializer.Serialize(response, options)}\n\n");
						await sw.FlushAsync(); // Send the text response back to the client.
					}
				}
			}
			catch (Exception e) { Debug.WriteLine($"{e}\n{e.Message}"); }
		}
		async Task Tokenize(ChatQuery query, HttpContext context) {
			var tokens = model.Tokenize(query.messages);
			var tokenizationResponse = new LocalTokenizeResponse();
			foreach (var token in tokens) { tokenizationResponse.Add(new(token, Model.vocab[token])); }

			await using var sw = new StreamWriter(context.Response.Body);
			await sw.WriteLineAsync(JsonSerializer.Serialize(tokenizationResponse, options));
			await sw.FlushAsync();
		}
	}

	record LocalResponse(string content, string delta, string finish_reason);
	class LocalTokenizeResponse : List<LocalTokenizeResponse.LocalToken> { public record LocalToken(int id, string str); }
}
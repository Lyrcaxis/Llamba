using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System;

namespace Llamba.Server {
    public class CompletionEndpoint {
        JsonSerializerOptions options = new();
        Model model => Model.instance;

        public CompletionEndpoint(WebApplication app) {
            app.MapPost("/text-completion/completion", async (context) => await Complete(await JsonSerializer.DeserializeAsync<CompletionQuery>(context.Request.Body), context));
        }

        async Task Complete(CompletionQuery query, HttpContext context) {
            try {
                //var msg = model.format.TurnToString(query.messages);
                await using var sw = new StreamWriter(context.Response.Body);

                var totalT = Model.instance.Tokenize(query.prompt).Count;
                if (totalT + query.max_tokens >= Model.instance.modelParams.ContextSize) {
                    await sw.WriteLineAsync($"Body of {totalT} requested {query.max_tokens}, surpassing {Model.instance.modelParams.ContextSize}.\n\n");
                    await sw.FlushAsync(); // Send the text response back to the client.
                    return;
                }

                using var request = Model.instance.AddRequest(query.prompt, query);
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
    }
}

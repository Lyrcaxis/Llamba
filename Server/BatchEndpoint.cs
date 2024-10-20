using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System;
using System.Text.Json.Serialization;
using System.Linq;
using System.Collections;

namespace Llamba.Server {
    namespace Llamba.Server {
        public class BatchEndpoint {
            JsonSerializerOptions options = new();

            public BatchEndpoint(WebApplication app) {
                app.MapPost("/batch", async (context) => await Batch(await JsonSerializer.DeserializeAsync<QueryBatch>(context.Request.Body), context));
            }

            async Task Batch(QueryBatch batchQuery, HttpContext context) {
                // Prettify the content of the requests
                for (int i = 0; i < (batchQuery.completionQueries?.Length ?? 0); i++) { batchQuery.completionQueries[i].prompt = batchQuery.completionQueries[i].prompt.Trim(); }
                for (int i = 0; i < (batchQuery.chatQueries?.Length ?? 0); i++) { foreach (var m in batchQuery.chatQueries[i].messages[..^1]) { m.content = m.content.Trim(); } }

                try {
                    await using var sw = new StreamWriter(context.Response.Body);

                    // Create the stop tokens once and share them among all InferenceRequest instances.
                    HashSet<int> stopTokens = (batchQuery.stop?.Any() == true) ? batchQuery.stop.Select(x => Model.instance.Tokenize(x.Replace("\\n", "\n"))[0]).ToHashSet() : null;

                    // Create the requests -- for both Chat and Completion queries.
                    List<InferenceRequest> requests = [];
                    for (int i = 0; i < (batchQuery.completionQueries?.Length ?? 0); i++) {
                        CompletionQuery query = batchQuery.completionQueries[i];
                        if (await SurpassesPromptLimits(Model.instance.Tokenize(query.prompt).Count, query)) { return; }
                        requests.Add(Model.instance.AddRequest(query.prompt, query, stopTokens));
                    }
                    for (int i = 0; i < (batchQuery.chatQueries?.Length ?? 0); i++) {
                        ChatQuery query = batchQuery.chatQueries[i];
                        if (await SurpassesPromptLimits(Model.instance.Tokenize(query.messages).Count, query)) { return; }
                        requests.Add(Model.instance.AddRequest(query, stopTokens));
                    }

                    // Map the response data to their IDs, preparing to send them back asynchronously, and keeping their state and index.
                    var queryCompletionMap = requests.Select(x => false).ToList();
                    var wholeResponses = requests.Select(x => "").ToList();
                    var batchResponse_stream = new BatchResponse() { responses = [] };

                    while (queryCompletionMap.Any(isComplete => !isComplete)) {
                        batchResponse_stream.responses.Clear();

                        // Gather the responses for this batch.
                        for (int i = 0; i < requests.Count; i++) {
                            if (queryCompletionMap[i]) { continue; }
                            if (GrabResponse(i) is not LocalResponse response) { continue; }
                            if (batchQuery.stream) { batchResponse_stream.responses.Add(new() { id = i, response = response }); }
                            wholeResponses[i] += response.delta;
                        }

                        // Send the query back to the client after retrieving this batch's responses.
                        if (batchQuery.stream && batchResponse_stream.responses.Count != 0) {
                            await sw.WriteLineAsync($"data: {JsonSerializer.Serialize(batchResponse_stream, options)}\n\n");
                            await sw.FlushAsync(); // Send the text response back to the client.
                        }

                        await Task.Delay(1);
                    }

                    // If we're not streaming, send a response that includes ALL completed text for the batch.
                    if (!batchQuery.stream) {
                        var wholeResponse = new BatchResponse() { responses = [] };
                        for (int i = 0; i < wholeResponses.Count; i++) {
                            var finish_reason = batchResponse_stream.responses[i].response.finish_reason;
                            wholeResponse.responses[i].id = i;
                            wholeResponse.responses[i].response = new(wholeResponses[i], default, finish_reason);
                        }
                        await sw.WriteLineAsync(JsonSerializer.Serialize(wholeResponse.responses, options));
                        await sw.FlushAsync();
                    }


                    // Tries to grab the response from specified request ID. Request is disposed if completed.
                    LocalResponse GrabResponse(int id) {
                        var request = requests[id];
                        if (!request.nextResponse.TryDequeue(out var r)) {
                            if (!request.needsGen) {
                                (request as IDisposable).Dispose();
                                queryCompletionMap[id] = true;
                            }
                            return null;
                        }
                        return new LocalResponse(default, r.response, r.stopReason);
                    }
                    async Task<bool> SurpassesPromptLimits(int tokenCount, IQueryParamsContainer query) {
                        if (tokenCount + query.max_tokens >= Model.instance.modelParams.ContextSize) {
                            await sw.WriteLineAsync($"Body of {tokenCount} requested {query.max_tokens}, surpassing {Model.instance.modelParams.ContextSize}.\n\n");
                            await sw.FlushAsync(); // Send the text response back to the client.
                            return true;
                        }
                        return false;
                    }
                }
                catch (Exception e) { Debug.WriteLine($"{e}\n{e.Message}"); }
            }
        }

        public class BatchResponse {
            public List<QueryResponse> responses { get; set; }

            public class QueryResponse {
                public int id { get; set; }
                public LocalResponse response { get; set; }
            }
        }

        public struct QueryBatch {
            public CompletionQuery[] completionQueries { get; set; }
            public ChatQuery[] chatQueries { get; set; }
            public string[] stop { get; set; }
            public bool stream { get; set; }
        }
    }
}
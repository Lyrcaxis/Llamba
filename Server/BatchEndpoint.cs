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
                foreach (var query in batchQuery.queries) {
                    foreach (var m in query.messages[..^1]) { m.content = m.content.Trim(); }
                    for (int i = 1; i < query.messages.Length; i++) { if (query.messages[i].role == "system") { query.messages[i].role = "next"; } }
                }

                try {
                    await using var sw = new StreamWriter(context.Response.Body);
                    HashSet<int> stopTokens = (batchQuery.stop?.Any() == true) ? batchQuery.stop.Select(x => Model.instance.Tokenize(x.Replace("\\n", "\n"))[0]).ToHashSet() : null;


                    // Create the requests
                    List<InferenceRequest> requests = new();
                    foreach (var query in batchQuery.queries) {
                        var totalT = Model.instance.Tokenize(query.messages).Count;
                        var maxT = Model.instance.modelParams.ContextSize;
                        if (totalT + query.max_tokens >= maxT) {
                            await sw.WriteLineAsync($"Body of {totalT} requested {query.max_tokens}, surpassing {maxT}.\n\n");
                            await sw.FlushAsync(); // Send the text response back to the client.
                            return;
                        }
                        requests.Add(Model.instance.AddRequest(query, stopTokens));
                    }

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
                        if (batchQuery.stream && batchResponse_stream.responses.Any()) {
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
                }
                catch (Exception e) { Debug.WriteLine($"{e}\n{e.Message}"); }
            }
        }

        class BatchResponse {
            public List<QueryResponse> responses { get; set; }

            public class QueryResponse {
                public int id { get; set; }
                public LocalResponse response { get; set; }
            }
        }

        public struct QueryBatch {
            public ChatQuery[] queries { get; set; }
            public string[] stop { get; set; }
            public bool stream { get; set; }
        }
    }
}
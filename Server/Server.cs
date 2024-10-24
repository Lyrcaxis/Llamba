﻿using Llamba.Server.Llamba.Server;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

using System;
using System.Linq;

namespace Llamba.Server {
    public class Server {
        public static ChatEndpoint chat { get; private set; }
        public static WebApplication app { get; private set; }

        public static void Create(string[] args) {
            var builder = WebApplication.CreateBuilder(args);
            builder.WebHost.UseUrls("http://localhost:5059");
            app = builder.Build();
            app.UseDefaultFiles().UseStaticFiles();
            Init();

            app.Start();
            LoadModel();
            NotifyServerStartup();
            app.WaitForShutdown();
        }

        static void Init() {
            app.MapGet("/ping", async (context) => { context.Response.StatusCode = StatusCodes.Status200OK; });
            chat = new ChatEndpoint(app);
            _ = new CompletionEndpoint(app);
            _ = new ClassificationEndpoint(app);
            _ = new BatchEndpoint(app);
            //_ = new Tests.BatchingTest(app);
        }

        static Model LoadModel() {
            string modelPath = ""; // Replace this if you don't want to be asked for the model path each startup.
            if (string.IsNullOrWhiteSpace(modelPath)) {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("Insert gguf file path:");
                Console.ForegroundColor = ConsoleColor.White;
                modelPath = Console.ReadLine();
            }

            if (modelPath.ToLower().Contains("ministral")) { return new Model(modelPath, new MistralFormat()); }
            if (modelPath.ToLower().Contains("mistral")) { return new Model(modelPath, new MistralFormat()); }
            if (modelPath.ToLower().Contains("gemma-2")) { return new Model(modelPath, new Gemma2Format()); }
            if (modelPath.ToLower().Contains("llama-3")) { return new Model(modelPath, new LLama3Format()); }

            return new Model(modelPath, new ChatMLFormat());
        }

        static void NotifyServerStartup() {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("Server initialized at ");
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine($"{app.Services.GetService<IServer>().Features.Get<IServerAddressesFeature>().Addresses.LastOrDefault()}/");
            Console.ForegroundColor = ConsoleColor.Gray;
        }
    }
}

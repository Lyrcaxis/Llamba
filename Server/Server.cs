using Microsoft.AspNetCore.Builder;

using System;

namespace Llamba.Server {
	public class Server {
		public static ChatEndpoint chat { get; private set; }
		public static WebApplication app { get; private set; }

		public static void Create(string[] args) {
			var builder = WebApplication.CreateBuilder(args);
			app = builder.Build();
			app.UseDefaultFiles().UseStaticFiles();
			chat = new ChatEndpoint(app, LoadModel());
			NotifyServerStartup();
			//_ = new Tests.BatchingTest();
			app.Run();
		}

		static Model LoadModel() {
			string modelPath = ""; // Replace this if you don't want to be asked for the model path each startup.

			if (string.IsNullOrWhiteSpace(modelPath)) {
				Console.ForegroundColor = ConsoleColor.Yellow;
				Console.WriteLine("Insert gguf file path:");
				Console.ForegroundColor = ConsoleColor.White;
				modelPath = Console.ReadLine();
			}

			return new Model(modelPath, new LLama3Format());
		}

		static void NotifyServerStartup() {
			Console.ForegroundColor = ConsoleColor.Green;
			Console.Write("Server initialized at ");
			Console.ForegroundColor = ConsoleColor.Blue;
			Console.WriteLine("http://localhost:5059/");
			Console.ForegroundColor = ConsoleColor.Gray;
		}
	}
}

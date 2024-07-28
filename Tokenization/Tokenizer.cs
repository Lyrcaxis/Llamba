using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;

namespace Llamba.Tokenization {
	/// <summary>
	/// A C# wrapper for huggingface's python tokenizer library. Can encode and decode <b>(tokens -> string -> tokens)</b> fluently on top-tier level.
	/// <para> Internally hosts a python server and the tokenization happens via localhost endpoint <b>requests -> responses</b>. See `tokenization.py`. </para>
	/// <para> This is slower than native encoding/decoding because of the serialization overhead, but fast enough for any and all purposes. </para>
	/// </summary>
	public class Tokenizer {

		static HttpClient client = new() { Timeout = TimeSpan.FromSeconds(5) };
		static Process currentServerProcess;
		const string hostIP = "http://127.0.0.1:8150";

		static Tokenizer() => InitializeTokenizer("Tokenization/llama-3-tokenizer.json");

		/// <summary> Load the tokenizer from the specified path. This internally creates the python server that hosts the tokenizer. </summary>
		/// <remarks> This method should be called each time a new tokenizer should be loaded. </remarks>
		public static void InitializeTokenizer(string tokenizerPath) {
			if (currentServerProcess != null) { currentServerProcess?.Kill(); }
			Console.WriteLine($"Starting python server at {hostIP} -- this will act as the tokenizer.");
			currentServerProcess = new Process { StartInfo = new("python", $"Tokenization/tokenization.py --path {tokenizerPath}") { UseShellExecute = false, CreateNoWindow = true } };
			currentServerProcess.Start();

			// Make sure the python application exits together with the server. NOTE: This will not occur when pressing 'STOP' from VS.
			AppDomain.CurrentDomain.ProcessExit += (s, e) => { if (!currentServerProcess.HasExited) { currentServerProcess.Kill(); } };
			var curProccess = Process.GetCurrentProcess(); curProccess.EnableRaisingEvents = true; curProccess.Exited += (_, _) => currentServerProcess.Kill();
		}


		/// <summary> Turns the string into a sequence of tokens. </summary>
		public static List<int> Encode(string text) => EncodeAsync(text).Result;

		/// <summary> Turns the sequence of tokens into a string. </summary>
		public static string Decode(List<int> tokens) => DecodeAsync(tokens).Result;

		/// <summary> Turns the string into a sequence of tokens. </summary>
		public async static Task<List<int>> EncodeAsync(string text) {
			var response = "";
			try {
				var message = new HttpRequestMessage(HttpMethod.Post, $"{hostIP}/encode") { Content = JsonContent.Create(new { text }) };
				using var httpResponse = await client.SendAsync(message);
				return JsonSerializer.Deserialize<List<int>>(response = await httpResponse.Content.ReadAsStringAsync());
			}
			catch { Debug.WriteLine($"Request failed: {response}"); return []; }
		}

		/// <summary> Turns the sequence of tokens into a string. </summary>
		public async static Task<string> DecodeAsync(List<int> tokens) {
			var response = "";
			try {
				var message = new HttpRequestMessage(HttpMethod.Post, $"{hostIP}/decode") { Content = JsonContent.Create(new { tokens }) };
				using var httpResponse = await client.SendAsync(message);
				return (response = await httpResponse.Content.ReadAsStringAsync())[1..^1]; // Quicker `Deserialize<string>(..)`
			}
			catch { Debug.WriteLine($"Request failed: {response}"); return ""; }
		}
	}
}

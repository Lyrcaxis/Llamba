# LLaMbA (Large Language Model Batching Application)
LLaMbA is a minimalistic cross-platform batching engine/server for LLMs, powered by [ASP.NET Core](https://dotnet.microsoft.com/en-us/apps/aspnet) and [LLamaSharp](https://github.com/SciSharp/LLamaSharp).

The engine's goal is to be able to serve multiple requests with small models as quick as possible, and it was made while having in mind its primary purposes of Serving, Classifying, and Generating Synthetic Data, within a minimal and extensible environment.

## Why is it fast
LLaMbA introduces quick and customizable ways to sample, made possible by .NET's `System.Numerics.Tensors` and threading. The Out-Of-The-Box sampling is arguably not as extensive as llama.cpp's, but it serves its purposes nicely and it's quite faster.

In addition, it utilizes llama.cpp's token grouping features to reduce the total amount of tokens in the batch, by reusing tokens that share the same position in multiple sequences.

## What it isn't
While LLaMbA contains a WebUI for chatting with the LLM, chatting on it will not be as efficient as backends created for single-user sessions. Reason is there is no per-request caching, in favor of re-processing multiple prompts together and grouping tokens among multiple sequences to improve overall output speed. That said, the primary use of the WebUI is testing any imposed changes, custom samplers, or systems.

It also might need some tweaking to work as an efficient serving backend (e.g.: switching to [AdditiveContextRefresh](https://github.com/Lyrcaxis/Llamba/blob/main/Batching/ContextRefresher.cs#L38)).

## Who it's intended for
Anyone can use LLaMbA for Synthetic Data generation locally as it is, but for more advanced purposes like Serving or Classifying, the primary target audience is Developers that should create safeguards (e.g. limits for max_tokens, potential moderation) and other systems to compliment the backend and take advantage of the high speeds. Developers are encouraged to experiment and customize the engine to their specs.

## Requirements
- .NET 8 SDK. Necessary for building and running the project.
- Python (+ packages). After installing python, install the necessary packages
```
pip install tokenizers uvicorn fastapi asyncio
```

## Videos + Tests
###### The model is LLama3.1-Instruct-8B-q8, and the tests happened on a single 4080 utilizing 16GB of VRAM.

#### Batching Test (w/ flash attention)

https://github.com/user-attachments/assets/d3316c94-b81f-439d-9c3d-0cf3e81dc951

#### WebUI - Made for Testing:

https://github.com/user-attachments/assets/bb06ccc1-b8f4-4370-b53e-f0786340e946

## [Example Usage](https://github.com/Lyrcaxis/Llamba/wiki/Example-Usage)
Visit [the wiki page](https://github.com/Lyrcaxis/Llamba/wiki/Example-Usage) for example usage of the API.

Intended usage is to make a bunch of back-to-back requests to the backend and await to parse them when they're ready.

Make sure to tweak the parameters in [`Model.cs`](https://github.com/Lyrcaxis/Llamba/blob/main/Model.cs) according to your GPU. The default parameters are for LLaMA3-8B with 16GB of VRAM.

![image](https://github.com/user-attachments/assets/368eec49-7942-4783-b7b9-b0eb68ac293a)

If you have more VRAM, you can increase the Context Size to be able to process more requests at the same time.

Enabling Flash Attention can (/will) also increase generation speed.

## Supported models
This supports all models currently supported by llama.cpp, but there are only built-in templates for LLama3 and ChatML prompt formats.

(see https://github.com/Lyrcaxis/Llamba/blob/main/InferenceFormat.cs to add your own prompt format)

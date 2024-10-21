# LLaMbA (Large Language Model Batching Application)
LLaMbA is a minimalistic cross-platform batching engine/server for LLMs, powered by [ASP.NET Core](https://dotnet.microsoft.com/en-us/apps/aspnet) and [LLamaSharp](https://github.com/SciSharp/LLamaSharp).

The engine's goal is to be able to serve multiple requests with small models as quick as possible, and it was made while having in mind its primary purposes of Serving, Classifying, and Generating Synthetic Data, within a minimal and extensible environment.



## Why is it fast
LLaMbA introduces quick and customizable ways to sample, made possible by .NET's `System.Numerics.Tensors` and threading. The Out-Of-The-Box sampling is arguably not as extensive as llama.cpp's, but it serves its purposes nicely and it's quite faster (up to ~10x increasing with smaller model sizes).

In addition, it hosts a python tokenizer, and utilizes llama.cpp's token grouping features to reduce the total amount of tokens in the batch, by reusing tokens that share the same position in multiple sequences, reducing the total amount of tokens the model sees. This can further be taken advantage of during multiple classifications of the same prompt, where most tokens are the same but the classification purposes change.



## What it isn't
While LLaMbA contains a basic Web UI for chatting with the LLM, it wasn't made to contain rich features and single-user session efficiency, but with ease-of-testing in mind. That said, the primary use of the Web UI is testing any imposed changes, custom samplers, or systems.

It also isn't an all-in-one & one-for-all deliverable; the user is expected to get hands-on and adjust code parts to their needs.



## Who it's intended for
Anyone can use LLaMbA for Synthetic Data generation locally as it is, but for more advanced purposes like Serving or Classifying, the primary target audience is Developers that should create safeguards (e.g. auth, limits for max_tokens, moderation) and other systems to compliment the backend and take advantage of the high speeds.

Developers are encouraged to experiment and customize the engine to their specs.



## Requirements
- [CUDA 12](https://developer.nvidia.com/cuda-12-0-0-download-archive) or the backend of your choice (CUDA11, CUDA12, Vulkan, OpenCL, Metal, CPU).
- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download). Necessary for building and running the project.
- [Python](https://www.python.org/downloads) (+ packages). After installing python, install the necessary packages:
```
pip install tokenizers uvicorn fastapi asyncio requests
```



## Videos
###### The model used in the videos is LLama3.1-Instruct-8B-Q8, on a single RTX 4080, utilizing ~12GB of VRAM.


#### Batching Test (w/ flash attention)

###### About double the speed in comparison to using the llama.cpp sampler.

https://github.com/user-attachments/assets/12c2845e-2a20-41df-99fe-fce641512cf0


#### WebUI - Made for Testing:

###### Chat UI supports basic back & forth functionality & message editing/deleting.

https://github.com/user-attachments/assets/11ecd164-5311-4047-bae0-16a8adef621c

###### Batches sent with Completion mode get passed without formatting, whereas Chat mode formats them to model's prompt format.

https://github.com/user-attachments/assets/a32f9bfb-f6ca-4c77-ab3e-8aef477a7093

###### It's easy and fast to navigate a model to generate a specific json field from your specs.

https://github.com/user-attachments/assets/ccaced6c-c47e-4c7b-87ae-dcaef508d31f



## General tips
Check out [the General Guide](https://github.com/Lyrcaxis/Llamba/wiki/General-Guide) and [Example Usage](https://github.com/Lyrcaxis/Llamba/wiki/Example-Usage) for example usage of the API and a quick code tour.

Context Size can be increased in [`Model.cs`](https://github.com/Lyrcaxis/Llamba/blob/main/Model.cs) to further increase throughput. The default parameters are for LLaMA3.1-8B-Q8 with ~12GB of VRAM.

Enabling Flash Attention will also increase generation throughput.



## Supported models
LLaMbA supports all language models currently supported by llama.cpp.
- see [InferenceFormat.cs](https://github.com/Lyrcaxis/Llamba/blob/main/InferenceFormat.cs) to add your own prompt format.
- and [Tokenizer.cs](https://github.com/Lyrcaxis/Llamba/blob/main/Tokenization/Tokenizer.cs) for adding a tokenizer. It's easy!

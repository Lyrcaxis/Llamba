import uvicorn, fastapi, sys, asyncio, threading, os, requests; from tokenizers import Tokenizer; app = fastapi.FastAPI()

@app.post("/encode")
async def encode(request: fastapi.Request):
    return tokenizer.encode((await request.json())['text'], add_special_tokens=False).ids;

@app.post("/decode")
async def decode(request: fastapi.Request):
    return tokenizer.decode((await request.json())['tokens'], False);

async def lifetime_loop():
    print("----- Tokenization server initialized -----");
    while True:
        try: requests.get("http://localhost:5059/ping"); print("PONG"); await asyncio.sleep(2);
        except: print("NVM.. quitting"); os._exit(1);

if __name__ == "__main__":
    print("Loading tokenizer from path: '" + sys.argv[sys.argv.index('--path') + 1] + "'");
    tokenizer: Tokenizer = Tokenizer.from_file(sys.argv[sys.argv.index('--path') + 1]);
    threading.Thread(target=asyncio.run, args=(lifetime_loop(),)).start()
    uvicorn.run(app, host=None, port=8150, log_level=None, timeout_keep_alive=5);
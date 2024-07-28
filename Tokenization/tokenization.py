import os, uvicorn, fastapi, sys, asyncio, time; from tokenizers import Tokenizer; app = fastapi.FastAPI()

@app.post("/encode")
async def encode(request: fastapi.Request):
    return tokenizer.encode((await request.json())['text'], add_special_tokens=False).ids;

@app.post("/decode")
async def decode(request: fastapi.Request):
    return tokenizer.decode((await request.json())['tokens'], False);

if __name__ == "__main__":
    tokenizer: Tokenizer = Tokenizer.from_file(sys.argv[sys.argv.index('--path') + 1]);
    loop = asyncio.get_event_loop(); asyncio.set_event_loop(loop);
    uvicorn.run(app, host=None, port=8150, log_level=None, timeout_keep_alive=5);
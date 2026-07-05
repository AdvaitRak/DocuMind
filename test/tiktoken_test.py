import tiktoken

try:
    enc = tiktoken.get_encoding("cl100k_base")
    print("✅ Tokenizer loaded")
    print(enc.encode("Hello world"))
except Exception as e:
    print(e)
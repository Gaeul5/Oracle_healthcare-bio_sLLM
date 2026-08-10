import tiktoken
text = "나는 LangChain을 좋아합니다."
encoding_cl100k = tiktoken.get_encoding("cl100k_base")
encoding_o200k = tiktoken.get_encoding("o200k_base")
print("문장:", text)
print("cl100k_base token 수:", len(encoding_cl100k.encode(text)))
print("o200k_base token 수:", len(encoding_o200k.encode(text)))
print("o200k token ids:", encoding_o200k.encode(text))
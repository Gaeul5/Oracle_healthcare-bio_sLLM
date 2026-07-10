import tiktoken
encoding = tiktoken.get_encoding("o200k_base")
examples = [
"I love LangChain",
"나는 랭체인을 좋아합니다",
"RAG = Retrieval Augmented Generation",
]
for text in examples:
    tokens = encoding.encode(text)
    print("=" * 50)
    print("문장:", text)
    print("토큰 수:", len(tokens))
    print("토큰 ID:", tokens)
    print("토큰 분해:")
    
for token in tokens:
    print(f"'{encoding.decode([token])}' ({token})")
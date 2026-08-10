import tiktoken
from langchain_text_splitters import TokenTextSplitter

encoding = tiktoken.get_encoding("o200k_base")

def num_tokens_from_string(string: str) -> int:
    return len(encoding.encode(string))

sample_text = "This splitter is essential for managing LLM context windows and optimizing API costs by counting tokens."
print(f"원본 텍스트의 총 토큰 수: {num_tokens_from_string(sample_text)}\n")

text_splitter = TokenTextSplitter(
chunk_size=15,
chunk_overlap=5,
encoding_name="o200k_base",
)
chunks = text_splitter.split_text(sample_text)
print(f"--- TokenTextSplitter 결과: 총 {len(chunks)}개---")
for i, chunk in enumerate(chunks):
    print(f"[{i+1}] (토큰: {num_tokens_from_string(chunk)}) {chunk}")
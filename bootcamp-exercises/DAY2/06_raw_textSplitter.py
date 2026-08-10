text = """
LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.
이 프레임워크의 핵심은 모듈화입니다.
주요 기능으로는 모델 I/O, Chains, RAG 등이 있습니다.
"""
chunk_size = 50
chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print(f"전체 글자 수: {len(text)}")
print(f"chunk_size: {chunk_size}")
print(f"chunk 개수: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"[{i+1}] {chunk}")
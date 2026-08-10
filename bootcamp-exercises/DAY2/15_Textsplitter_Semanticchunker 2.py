from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
load_dotenv()
sample_text = (
"태양은 별입니다. 달은 지구 주위를 돕니다. 화성은 태양계의 네 번째 행성입니다."
"RAG는 외부 문서를 검색해서 답변에 참고하는 방식입니다. "
"Embedding은 문장의 의미를 숫자 벡터로 바꾸는 과정입니다."
)
splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"))
chunks = splitter.split_text(sample_text)
print(f"--- SemanticChunker 결과: 총 {len(chunks)}개---")
for i, chunk in enumerate(chunks):
    print(f"[{i+1}] {chunk}")
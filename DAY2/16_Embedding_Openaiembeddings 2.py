from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

embeddings_model =OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

documents = [
    "저는 강아지를 좋아합니다.",
    "저는 고양이를 좋아합니다.",
    "저는 축구를 좋아합니다.",
]

document_embeddings = embeddings_model.embed_documents(documents)

query = "제가 좋아하는 동물은 무엇인가요?"
query_embedding = embeddings_model.embed_query(query)

print("문서 embedding 개수:", len(document_embeddings))
print("문서 embedding 차원:", len(document_embeddings[0]))
print("질문 embedding 차원:", len(query_embedding))
print("첫 번째 문서 embedding 앞 5개:",

document_embeddings[0][:5])
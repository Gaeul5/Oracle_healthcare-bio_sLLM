from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_NAME = "day5_pgvector_tool_demo"
FILE_PATH = Path(__file__).resolve().parent / "data" / "SPRi AI Brief_10월호_산업동향_1002_F.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"PDF 파일을 찾을 수 없습니다: {FILE_PATH}\n"
        "실습용 PDF를 ./data 폴더에 준비하세요."
    )

embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

db_config = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "testdb"),
    "user": os.getenv("POSTGRES_USER", "test"),
    "password": os.getenv("POSTGRES_PASSWORD", "5748"),
}

connection_string = PGVector.connection_string_from_db_params(driver="psycopg", **db_config)

print("=" * 80)
print("01. PGVector.from_documents()로 VectorStore 만들기")
print("=" * 80)

loader = PyPDFLoader(str(FILE_PATH))
pages = loader.load()
print(f"\n[PDF 로드 완료] page 수: {len(pages)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
documents = text_splitter.split_documents(pages)
print(f"[문서 분할 완료] chunk 개수: {len(documents)}")

embeddings_model = OpenAIEmbeddings(model=embedding_model_name)

vectorstore = PGVector.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name=COLLECTION_NAME,
    connection=connection_string,
    pre_delete_collection=True,
)

print("\n[PGVector 저장 완료]")
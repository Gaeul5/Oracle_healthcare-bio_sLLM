from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

QUESTION = "LangChain에서 rag는 왜 필요한가요?"
TOP_K = 1


def env(name, default=""):
    return os.getenv(name, default).strip()


def connect_db():
    return psycopg.connect(
        host=env("POSTGRES_HOST", "localhost"),
        port=int(env("POSTGRES_PORT", "5432")),
        dbname=env("POSTGRES_DB"),
        user=env("POSTGRES_USER"),
        password=env("POSTGRES_PASSWORD"),
        row_factory=dict_row,
    )


def to_pgvector(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"


print("질문:", QUESTION)

embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(model=embedding_model)
query_vector = embeddings.embed_query(QUESTION)
query_vector_text = to_pgvector(query_vector)

conn = connect_db()

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT
            d.file_name,
            c.chunk_index,
            c.page_number,
            c.content,
            e.embedding <=> %s::vector AS distance
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        JOIN rag_documents d ON c.document_id = d.id
        WHERE d.file_type = 'txt'
        AND e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vector_text, embedding_model, query_vector_text, TOP_K),
    )
    results = cur.fetchall()

conn.close()

print("\n검색 결과")
for i, row in enumerate(results, start=1):
    print("=" * 80)
    print(f"[{i}] distance: {row['distance']:.4f}")
    print("file:", row["file_name"])
    print("chunk_index:", row["chunk_index"])
    print(row["content"][:500])

print("\n다음 단계: python 04_insert_pdf_embeddings.py")
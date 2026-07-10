from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
load_dotenv()

TXT_PATH = DATA_DIR / "Sample_Text.txt"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


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


print("[1] TXT 읽기")
text = TXT_PATH.read_text(encoding="utf-8")
print("글자 수:", len(text))

print("\n[2] chunk 만들기")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_text(text)
print("chunk 개수:", len(chunks))

print("\n[3] embedding 만들기")
embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(model=embedding_model)
vectors = embeddings.embed_documents(chunks)
print("embedding 개수:", len(vectors))
print("embedding 차원:", len(vectors[0]))

print("\n[4] DB에 저장")
conn = connect_db()

with conn.cursor() as cur:
    # 같은 TXT를 다시 실행할 때 중복 저장되지 않도록 기존 데이터를 삭제합니다.
    # rag_documents를 지우면 연결된 chunks, embeddings도 CASCADE로 같이 삭제됩니다.
    cur.execute("DELETE FROM rag_documents WHERE file_name = %s", (TXT_PATH.name,))

    cur.execute(
        """
        INSERT INTO rag_documents (title, file_name, file_type, page_count)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        ("LangChain Sample Text", TXT_PATH.name, "txt", None),
    )
    document_id = cur.fetchone()["id"]
    print("document_id:", document_id)

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        cur.execute(
            """
            INSERT INTO rag_chunks (document_id, chunk_index, page_number, content)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (document_id, i, None, chunk),
        )
        chunk_id = cur.fetchone()["id"]

        cur.execute(
            """
            INSERT INTO rag_embeddings (chunk_id, embedding_model, embedding)
            VALUES (%s, %s, %s::vector)
            """,
            (chunk_id, embedding_model, to_pgvector(vector)),
        )

        print(f"저장 완료: chunk_index={i}, chunk_id={chunk_id}")

conn.commit()
conn.close()

print("\nTXT 저장 완료")
print("다음 단계: python 03_search_txt_from_db.py")
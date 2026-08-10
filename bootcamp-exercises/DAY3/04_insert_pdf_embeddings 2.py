from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
load_dotenv()

PDF_PATH = DATA_DIR / "SPRi AI Brief_10월호_산업동향_1002_F.pdf"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


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


print("[1] PDF 로드")
loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()
print("페이지 수:", len(pages))

print("\n[2] PDF chunk 만들기")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(pages)

#max_chunks = int(env("MAX_CHUNKS_FOR_PRACTICE", "0") or "0")
#if max_chunks > 0:
#    print(f"일부 chunk만 사용: {max_chunks}개")
#    chunks = chunks[:max_chunks] 

print("chunk 개수:", len(chunks))

print("\n[3] embedding 만들기")
texts = [doc.page_content for doc in chunks]
embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(model=embedding_model)
vectors = embeddings.embed_documents(texts)
print("embedding 개수:", len(vectors))
print("embedding 차원:", len(vectors[0]))

print("\n[4] DB에 저장")
conn = connect_db()

with conn.cursor() as cur:
    # 같은 PDF를 다시 실행할 때 중복 저장되지 않도록 기존 데이터를 삭제합니다.
    cur.execute("DELETE FROM rag_documents WHERE file_name = %s", (PDF_PATH.name,))

    cur.execute(
        """
        INSERT INTO rag_documents (title, file_name, file_type, page_count)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        ("SPRi AI Brief 2025년 10월호", PDF_PATH.name, "pdf", len(pages)),
    )
    document_id = cur.fetchone()["id"]
    print("document_id:", document_id)

    for i, (doc, vector) in enumerate(zip(chunks, vectors)):
        raw_page = doc.metadata.get("page")
        page_number = raw_page + 1 if isinstance(raw_page, int) else None

        cur.execute(
            """
            INSERT INTO rag_chunks (document_id, chunk_index, page_number, content)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (document_id, i, page_number, doc.page_content),
        )
        chunk_id = cur.fetchone()["id"]

        cur.execute(
            """
            INSERT INTO rag_embeddings (chunk_id, embedding_model, embedding)
            VALUES (%s, %s, %s::vector)
            """,
            (chunk_id, embedding_model, to_pgvector(vector)),
        )

        if i < 5 or i == len(chunks) - 1:
            print(f"저장 완료: chunk_index={i}, page={page_number}, chunk_id={chunk_id}")
        elif i == 5:
            print("... 중간 출력 생략 ...")

conn.commit()
conn.close()

print("\nPDF 저장 완료")
print("다음 단계: python 05_sql_vector_search.py")
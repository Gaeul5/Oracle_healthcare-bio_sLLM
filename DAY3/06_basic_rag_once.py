from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

QUESTION = "중국 AI 플러스 정책의 핵심 영역과 목표를 요약해줘."
TOP_K = 4

SYSTEM_PROMPT = """너는 문서 기반으로 답변하는 RAG 챗봇입니다.

규칙:
1. 반드시 제공된 context를 우선해서 답변하세요.
2. context에 없는 내용은 추측하지 말고 모른다고 말하세요.
3. 답변은 한국어로 작성하세요.
"""


class RagAnswer(BaseModel):
    answer: str = Field(description="context를 근거로 작성한 한국어 답변")

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

def search_chunks(question):
    embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    query_vector = embeddings.embed_query(question)
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
            WHERE d.file_type = 'pdf'
              AND e.embedding_model = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector_text, embedding_model, query_vector_text, TOP_K),
        )
        results = cur.fetchall()
    conn.close()
    return results

def format_context(results):
    context = ""
    for i, row in enumerate(results, start=1):
        context += f"[문서 {i} / {row['file_name']} / p.{row['page_number']}]\n"
        context += row["content"] + "\n\n"
    return context

def generate_answer(question, results):
    context = format_context(results)
    parser = PydanticOutputParser(pydantic_object=RagAnswer)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT
                + "\n반드시 아래 출력 형식을 지키세요.\n{format_instructions}",
            ),
            (
                "human",
                """아래 context만 참고해서 질문에 답변하세요.

[context]
{context}

[질문]
{question}
""",
            ),
        ]
    )

    llm = ChatOpenAI(model=env("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    chain = prompt | llm | parser
    result = chain.invoke(
        {
            "context": context,
            "question": question,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return result.answer


print("질문:", QUESTION)

results = search_chunks(QUESTION)

print("\n[검색된 문서]")
for i, row in enumerate(results, start=1):
    print(f"{i}. {row['file_name']} / p.{row['page_number']} / distance={row['distance']:.4f}")
    print(row["content"][:300].replace("\n", " "))
    print()

answer = generate_answer(QUESTION, results)

print("=" * 80)
print("답변")
print("=" * 80)
print(answer)

print("\n다음 단계: python 07_cli_rag_chatbot.py")
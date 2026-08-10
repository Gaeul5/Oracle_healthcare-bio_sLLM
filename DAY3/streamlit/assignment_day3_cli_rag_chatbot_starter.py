from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TOP_K = 4

SYSTEM_PROMPT = """너는 문서 기반으로 답변하는 RAG 챗봇입니다.

규칙:
1. 반드시 제공된 context를 우선해서 답변하세요.
2. context에 없는 내용은 추측하지 말고 모른다고 말하세요.
3. 답변은 한국어로 작성하세요.
"""


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


def retrieve_documents(question):
    """TODO 1: 질문을 embedding하고 DB에서 가까운 chunk를 검색하세요."""

    # 1. embedding 모델 준비
    embedding_model = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # TODO: 질문을 embedding하세요.
    # query_vector = ...
    # query_vector_text = to_pgvector(query_vector)

    # 2. DB 연결
    conn = connect_db()

    # TODO: rag_embeddings, rag_chunks, rag_documents를 JOIN해서 vector 검색 SQL을 실행하세요.
    # 힌트: ORDER BY e.embedding <=> %s::vector LIMIT %s
    # 결과 컬럼: file_name, chunk_index, page_number, content, distance

    # with conn.cursor() as cur:
    #     cur.execute("""
    #         SELECT ...
    #     """, (...))
    #     results = cur.fetchall()

    conn.close()

    # TODO: results를 반환하세요.
    raise NotImplementedError("retrieve_documents()의 TODO를 완성하세요.")


def format_context(results):
    context = ""
    for i, row in enumerate(results, start=1):
        context += f"[문서 {i} / {row['file_name']} / p.{row['page_number']}]\n"
        context += row["content"] + "\n\n"
    return context


def generate_answer(question, results):
    """TODO 2: 검색 결과를 context로 만들고 LLM에게 답변을 요청하세요."""

    # TODO: 검색 결과를 context로 변환하세요.
    # context = ...

    # TODO: user_prompt를 만드세요.
    # user_prompt = f"""..."""

    # TODO: ChatOpenAI를 만들고 invoke하세요.
    # llm = ChatOpenAI(...)
    # response = llm.invoke([...])

    # TODO: response.content를 반환하세요.
    raise NotImplementedError("generate_answer()의 TODO를 완성하세요.")


def print_sources(results):
    print("\n[검색된 출처]")
    for i, row in enumerate(results, start=1):
        print(f"{i}. {row['file_name']} / p.{row['page_number']} / distance={row['distance']:.4f}")


print("Day 3 과제: CLI RAG 챗봇")
print("종료하려면 quit, exit, 종료 중 하나를 입력하세요.\n")

while True:
    question = input("질문 > ").strip()

    if not question:
        continue

    if question.lower() in {"quit", "exit"} or question == "종료":
        print("챗봇을 종료합니다.")
        break

    try:
        results = retrieve_documents(question)
        answer = generate_answer(question, results)
    except NotImplementedError as e:
        print(e)
        break
    except Exception as e:
        print("오류가 발생했습니다.")
        print(e)
        continue

    print("\n" + "=" * 80)
    print("답변")
    print("=" * 80)
    print(answer)
    print_sources(results)
    print()
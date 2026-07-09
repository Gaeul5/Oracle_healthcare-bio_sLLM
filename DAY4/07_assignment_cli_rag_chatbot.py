"""Day 4 과제: Dense/Sparse/Hybrid 전환이 되는 CLI RAG 챗봇.

Day 3에서 만든 CLI 챗봇(질문 -> 검색 -> LLM 답변 반복 루프)에
Day 4에서 만든 dense/sparse/hybrid retriever를 연결합니다.

실행 예시:
    python DAY4/07_assignment_cli_rag_chatbot.py

완성 조건:
- 실행 중 "/mode dense", "/mode sparse", "/mode hybrid" 명령으로 검색 방식을 바꿀 수 있다.
- 질문을 입력하면 현재 모드로 검색한 뒤, 검색 결과를 context로 LLM이 답변한다.
- 답변 아래에 검색된 출처(파일명, 페이지, 점수, matched_by)를 출력한다.
- quit / exit / 종료 로 반복문을 빠져나간다.

아래 dense_search / sparse_search / hybrid_search / retrieve 는
05_hybrid_retriever.py, 06_retriever_rag_once.py에서 이미 완성한 코드를 그대로 옮겨왔습니다.
TODO로 표시된 부분만 채우면 됩니다.
"""

import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from psycopg2.extras import RealDictCursor

load_dotenv()

DEFAULT_MODE = "hybrid"  # dense, sparse, hybrid 중 하나
VALID_MODES = {"dense", "sparse", "hybrid"}

SYSTEM_PROMPT = """너는 문서 기반으로 답변하는 RAG 챗봇입니다.

규칙:
1. 반드시 제공된 context를 우선해서 답변하세요.
2. context에 없는 내용은 추측하지 말고 모른다고 말하세요.
3. 답변은 한국어로 작성하세요.
"""


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"환경변수 {name} 값이 없습니다. .env 파일을 확인하세요.")
    return value or ""


def connect_db():
    return psycopg2.connect(
        host=env("POSTGRES_HOST", "localhost"),
        port=env("POSTGRES_PORT", "5432"),
        dbname=env("POSTGRES_DB", required=True),
        user=env("POSTGRES_USER", required=True),
        password=env("POSTGRES_PASSWORD", required=True),
    )


def get_embedding_model_name() -> str:
    return env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=get_embedding_model_name())


def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=env("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


def get_top_k(default: int = 4) -> int:
    raw = env("TOP_K", str(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in values) + "]"


def fetch_all(conn, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def dense_search(conn, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
    embeddings = get_embeddings()
    query_vector = to_pgvector(embeddings.embed_query(query))
    embedding_model = get_embedding_model_name()

    sql = """
        SELECT
            c.id AS chunk_id,
            d.file_name,
            c.page_number,
            c.content,
            e.embedding <=> %s::vector AS distance,
            'dense' AS retriever_type
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        JOIN rag_documents d ON c.document_id = d.id
        WHERE e.embedding_model = %s
        ORDER BY distance ASC
        LIMIT %s;
    """
    rows = fetch_all(conn, sql, (query_vector, embedding_model, top_k))
    for row in rows:
        row["score_label"] = "distance"
        row["score"] = row.get("distance")
        row["matched_by"] = ["dense"]
    return rows


def sparse_search(conn, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    sql = """
        SELECT
            c.id AS chunk_id,
            d.file_name,
            c.page_number,
            c.content,
            ts_rank_cd(
                to_tsvector('simple', coalesce(c.content, '')),
                websearch_to_tsquery('simple', %s)
            ) AS sparse_score,
            'sparse' AS retriever_type
        FROM rag_chunks c
        JOIN rag_documents d ON c.document_id = d.id
        WHERE to_tsvector('simple', coalesce(c.content, ''))
              @@ websearch_to_tsquery('simple', %s)
        ORDER BY sparse_score DESC
        LIMIT %s;
    """
    rows = fetch_all(conn, sql, (query, query, top_k))
    for row in rows:
        row["score_label"] = "sparse_score"
        row["score"] = row.get("sparse_score")
        row["matched_by"] = ["sparse"]
    return rows


def hybrid_search(conn, query: str, *, top_k: int = 4, rrf_k: int = 60) -> list[dict[str, Any]]:
    pool_k = max(top_k * 2, top_k)
    dense_rows = dense_search(conn, query, top_k=pool_k)
    sparse_rows = sparse_search(conn, query, top_k=pool_k)

    merged: dict[int, dict[str, Any]] = {}

    def add_rows(rows: list[dict[str, Any]], source: str):
        for rank, row in enumerate(rows, start=1):
            chunk_id = int(row["chunk_id"])
            if chunk_id not in merged:
                item = dict(row)
                item["retriever_type"] = "hybrid"
                item["hybrid_score"] = 0.0
                item["matched_by"] = []
                merged[chunk_id] = item
            merged[chunk_id]["hybrid_score"] += 1.0 / (rrf_k + rank)
            if source not in merged[chunk_id]["matched_by"]:
                merged[chunk_id]["matched_by"].append(source)

    add_rows(dense_rows, "dense")
    add_rows(sparse_rows, "sparse")

    results = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
    for row in results:
        row["score_label"] = "hybrid_score"
        row["score"] = row.get("hybrid_score")
    return results


def retrieve(conn, query: str, *, mode: str, top_k: int) -> list[dict[str, Any]]:
    """TODO 1: mode("dense"/"sparse"/"hybrid")에 맞는 검색 함수를 호출하세요.

    힌트: 위에서 정의한 dense_search / sparse_search / hybrid_search 를 그대로 호출하면 됩니다.
    """
    retriever_type = mode.strip().lower()
    if retriever_type == "dense":
        return dense_search(conn, query, top_k=top_k)
    if retriever_type == "sparse":
        return sparse_search(conn, query, top_k=top_k)
    if retriever_type == "hybrid":
        return hybrid_search(conn, query, top_k=top_k)
    raise NotImplementedError("retrieve()의 TODO를 완성하세요.")


def format_context(results: list[dict[str, Any]]) -> str:
    """TODO 2: 검색 결과 리스트를 LLM에게 줄 context 문자열로 만드세요.

    힌트:
    - row["file_name"], row["page_number"], row["content"] 를 사용하세요.
    - row["matched_by"] (리스트)를 "+"로 join하면 "dense+sparse" 처럼 어떤 검색으로 걸렸는지 보여줄 수 있습니다.
    - 여러 문서를 구분자("\\n\\n---\\n\\n" 등)로 이어붙이세요.
    """
    parts: list[str] = []
    for i, row in enumerate(results, start=1):
        matched = "+".join(row.get("matched_by") or [row.get("retriever_type", "")])
        parts.append(
            f"[문서 {i} / {row.get('file_name')} / p.{row.get('page_number')} / {matched}]\n"
            f"{row.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, results: list[dict[str, Any]]) -> str:
    """TODO 3: context를 만들고 LLM에게 답변을 요청하세요.

    힌트:
    - results가 비어 있으면 LLM을 부르지 말고 안내 문구를 바로 반환하세요.
    - ChatPromptTemplate.from_template(...) 으로 SYSTEM_PROMPT + context + question을 넣은 프롬프트를 만드세요.
    - chain = prompt | get_llm() | StrOutputParser()
    - chain.invoke({"context": ..., "question": ...}) 로 답변 문자열을 받으세요.
    """
    if not results:
        return "문서에서 충분한 근거를 찾지 못했습니다."
    prompt = ChatPromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\ncontext:\n{context}\n\nquestion:\n{question}"
    )
    chain = prompt | get_llm() | StrOutputParser()
    context = format_context(results)
    return chain.invoke({"context": context, "question": question})


def print_sources(results: list[dict[str, Any]]) -> None:
    print("\n[검색된 출처]")
    if not results:
        print("- 검색된 출처가 없습니다.")
        return
    for i, row in enumerate(results, start=1):
        label = row.get("score_label", "score")
        score = row.get("score")
        score_text = f"{label}={score:.4f}" if isinstance(score, (int, float)) else f"{label}={score}"
        matched = "+".join(row.get("matched_by") or [row.get("retriever_type", "")])
        print(f"{i}. {row.get('file_name')} / p.{row.get('page_number')} / {matched} / {score_text}")


def parse_command(user_input: str, current_mode: str) -> tuple[str | None, str]:
    """TODO 4: "/mode dense" 같은 입력을 처리하세요.

    - user_input이 "/mode xxx" 형태이면:
        - xxx가 VALID_MODES 안에 있으면 새 모드를 반환하고, 없으면 안내 메시지를 출력한 뒤 current_mode를 유지하세요.
    - "/mode"로 시작하지 않으면 그대로 current_mode를 반환하세요.
    - 반환값: (다음에 검색할 때 쓸 mode, 그대로 실행할 질문 텍스트)
      단, "/mode ..." 명령 자체는 질문이 아니므로 두 번째 값은 빈 문자열("")로 반환하세요.

    이 함수는 아래 main() 루프에서
        mode, question = parse_command(user_input, mode)
        if not question:
            continue
    형태로 사용할 것을 가정합니다.
    """
    if not user_input.startswith("/mode"):
        # 명령어가 아니면 그대로 질문으로 취급
        return current_mode, user_input

    parts = user_input.split()  # 예: "/mode dense" -> ["/mode", "dense"]

    if len(parts) < 2 or parts[1] not in VALID_MODES:
        print(f"사용법: /mode dense|sparse|hybrid (현재 모드 유지: {current_mode})")
        return current_mode, ""

    new_mode = parts[1]
    print(f"모드를 '{new_mode}'로 변경했습니다.")
    return new_mode, ""
    raise NotImplementedError("parse_command()의 TODO를 완성하세요.")


def main():
    conn = connect_db()
    top_k = get_top_k(4)
    mode = DEFAULT_MODE

    print("Day 4 과제: Dense/Sparse/Hybrid CLI RAG 챗봇")
    print("모드 전환: /mode dense | /mode sparse | /mode hybrid")
    print("종료: quit, exit, 종료\n")
    print(f"현재 모드: {mode}\n")

    while True:
        user_input = input(f"[{mode}] 질문 > ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"} or user_input == "종료":
            print("챗봇을 종료합니다.")
            break

        try:
            mode, question = parse_command(user_input, mode)
        except NotImplementedError as e:
            print(e)
            break

        if not question:
            continue

        try:
            results = retrieve(conn, question, mode=mode, top_k=top_k)
            answer = generate_answer(question, results)
        except NotImplementedError as e:
            print(e)
            break
        except Exception as e:
            print("오류가 발생했습니다.")
            print(e)
            continue

        print("\n" + "=" * 80)
        print(f"답변 (mode={mode})")
        print("=" * 80)
        print(answer)
        print_sources(results)
        print()

    conn.close()


if __name__ == "__main__":
    main()

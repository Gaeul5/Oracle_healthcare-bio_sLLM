from __future__ import annotations

from backend.app.core.config import settings


def search_similar_chunks(
    question: str, user_id: int, top_k: int = 4, document_id: int | None = None
) -> list[dict]:
    """질문과 유사한 chunk를 PostgreSQL pgvector로 검색합니다. user_id가 소유한 문서에서만 검색합니다.

    TODO:
    1. 질문을 embedding합니다.
    2. repository.search_similar_chunks()를 호출합니다.
    3. 검색 결과 list[dict]를 반환합니다.
    """
    embedding_model = settings.OPENAI_EMBEDDING_MODEL
    # 1. 질문을 embedding합니다.

    from backend.app.services.embedding_service import embed_text
    query_embedding = embed_text(question)

    # 2. repository.search_similar_chunks()를 호출합니다.
    from backend.app.db.repository import search_similar_chunks
    rows = search_similar_chunks(
        query_embedding, embedding_model, top_k, user_id=user_id, document_id=document_id
    )

    # 3. 검색 결과 list[dict]를 반환합니다.
    return rows
    


def build_context(rows: list[dict]) -> str:
    """검색된 chunk 목록을 LLM에 넣을 context 문자열로 바꿉니다.

    TODO:
    - 각 row의 content를 모아 context 문자열을 만듭니다.
    - 출처 번호와 함께 넣으면 디버깅에 좋습니다.
    """
    context = ""
    for i, row in enumerate(rows):
        context += f"[{i+1}] {row['content']}\n"
    return context
    
def build_sources(rows: list[dict]) -> list[dict]:
    """검색 결과 row에서 프론트에 보여줄 sources 배열을 만듭니다.

    TODO:
    - document_id, document_title, file_name, page_number, chunk_index, preview 등을 구성합니다.
    - preview는 content 앞부분 100~150자 정도만 사용합니다.
    """
    sources = []
    for row in rows:
        source = {
            "document_id": row["document_id"],
            "document_title": row["document_title"],
            "file_name": row["file_name"],
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "preview": row["content"][:150] + "..." if len(row["content"]) > 150 else row["content"]
        }
        sources.append(source)
    return sources
    


def generate_answer(question: str, context: str) -> str:
    """LangChain Prompt + ChatOpenAI로 답변을 생성합니다.

    TODO:
    - ChatPromptTemplate을 만듭니다.
    - ChatOpenAI(model=settings.OPENAI_MODEL)를 생성합니다.
    - prompt | llm | StrOutputParser() 형태의 LCEL chain을 만듭니다.
    - question, context를 전달해 답변 문자열을 반환합니다.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    prompt_template = """너는 문서 기반 질의응답 챗봇이다.
아래 [문서 내용]만 참고해서 사용자의 질문에 답변해라.
문서 내용에서 답을 찾을 수 없으면 "등록된 문서에서 답을 찾을 수 없습니다."라고 답변해라.
답변은 한국어로 작성해라.
출처는 직접 만들어내지 마라.

[문서 내용]
{context}

[사용자 질문]
{question}
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer

   

def ask(question: str, user_id: int, top_k: int = 4) -> dict:
    """RAG 전체 흐름을 실행합니다.

    TODO:
    1. search_similar_chunks(question, top_k)
    2. 검색 결과가 없으면 안내 메시지 반환
    3. build_context(rows)
    4. generate_answer(question, context)
    5. build_sources(rows)
    6. {"answer": answer, "sources": sources} 반환
    """

    search_results = search_similar_chunks(question, user_id=user_id, top_k=top_k)
    if not search_results:
        return {"answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다.", "sources": []}
    
    context = build_context(search_results)
    answer = generate_answer(question, context)
    sources = build_sources(search_results)
    return {"answer": answer, "sources": sources}

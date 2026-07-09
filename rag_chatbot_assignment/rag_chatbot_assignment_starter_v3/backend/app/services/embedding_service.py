from __future__ import annotations

from backend.app.core.config import settings


def get_embedding_model():
    """OpenAIEmbeddings 객체를 생성합니다.

    TODO:
    - langchain_openai의 OpenAIEmbeddings를 import합니다.
    - model에는 settings.OPENAI_EMBEDDING_MODEL을 사용합니다.
    - 반환된 embedding 차원은 기존 DB의 vector(1536)과 맞아야 합니다.

    힌트:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    """문자열 하나를 embedding vector로 변환합니다.

    TODO:
    - get_embedding_model()로 embeddings 객체를 가져옵니다.
    - embeddings.embed_query(text)를 호출합니다.
    """
    embeddings = get_embedding_model()
    return embeddings.embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """문서 chunk 여러 개를 embedding vector 목록으로 변환합니다.

    TODO:
    - embeddings.embed_documents(texts)를 사용합니다.
    """
    embeddings = get_embedding_model()
    return embeddings.embed_documents(texts)

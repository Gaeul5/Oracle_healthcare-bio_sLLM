from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_PDF_PATH = DATA_DIR / "SPRi AI Brief_10월호_산업동향_1002_F.pdf"

DEFAULT_COLLECTION_NAME = "lc_retriever_dense_demo"
ENSEMBLE_COLLECTION_NAME = "lc_retriever_ensemble_demo"


def get_db_config() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "testdb"),
        "user": os.getenv("POSTGRES_USER", "test"),
        "password": os.getenv("POSTGRES_PASSWORD", "5748"),
    }


def get_connection_string() -> str:
    """LangChain PGVector가 사용하는 connection string을 만듭니다."""
    return PGVector.connection_string_from_db_params(
        driver="psycopg",
        **get_db_config(),
    )


def connect_db():
    """테이블 확인용 psycopg 연결입니다."""
    cfg = get_db_config()
    return psycopg.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
    )


def get_embedding_model_name() -> str:
    return (
        os.getenv("OPENAI_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=get_embedding_model_name())


def load_and_split_pdf(
    file_path: str | Path = DEFAULT_PDF_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """PDF를 읽고 LangChain Document chunk 목록으로 분리합니다."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다: {path}\n"
            "data 폴더에 실습 PDF를 넣고 다시 실행하세요."
        )

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.split_documents(pages)
    return documents


def print_documents(results: Iterable[Document], title: str = "검색 결과") -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)

    for idx, doc in enumerate(results, 1):
        metadata = doc.metadata or {}
        print(f"\n[결과 {idx}]")
        print(f"내용: {doc.page_content[:220].replace(chr(10), ' ')}...")
        print(f"출처: {metadata.get('source', metadata.get('file_name', '알 수 없음'))}")
        print(f"페이지: {metadata.get('page', metadata.get('page_number', '알 수 없음'))}")
        print("-" * 80)
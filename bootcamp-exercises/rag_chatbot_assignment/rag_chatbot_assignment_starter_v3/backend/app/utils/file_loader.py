from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def validate_file_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"지원하지 않는 파일 형식입니다. 허용 확장자: {allowed}")


def load_file_as_documents(file_path: str):
    """업로드 파일을 LangChain Document 목록으로 읽습니다.

    TODO:
    1. Path(file_path).suffix로 확장자를 확인합니다.
    2. .pdf 파일이면 PyPDFLoader를 사용합니다.
    3. .txt, .md 파일이면 TextLoader를 사용합니다.
    4. loader.load() 결과를 반환합니다.

    힌트:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            return loader.load()

        if suffix in [".txt", ".md"]:
            loader = TextLoader(str(path), encoding="utf-8")
            return loader.load()
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

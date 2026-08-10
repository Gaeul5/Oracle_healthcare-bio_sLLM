from pathlib import Path


def load_text(path: str) -> str:
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8")


def preview_text(text: str, limit: int = 300) -> str:
    return text[:limit]


class KnowledgeBase:
    def __init__(self, name: str):
        self.name = name
        self.documents = []

    def add_document(self, text: str) -> None:
        self.documents.append(text)

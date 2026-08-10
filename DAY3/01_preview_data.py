from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
print(DATA_DIR)
TXT_PATH = DATA_DIR / "Sample_Text.txt"
PDF_PATH = DATA_DIR / "SPRi AI Brief_10월호_산업동향_1002_F.pdf"

print("[1] data 폴더 파일")
for path in DATA_DIR.iterdir():
    print("-", path.name)

print("\n[2] TXT 미리보기")
text = TXT_PATH.read_text(encoding="utf-8")
print("글자 수:", len(text))
print(text[:500])

print("\n[3] PDF 미리보기")
loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()
print("PDF 페이지 수:", len(pages))
print("첫 페이지 metadata:", pages[0].metadata)
print("첫 페이지 내용 일부:")
print(pages[0].page_content[:500])

print("\n다음 단계: python 02_insert_txt_embeddings.py")
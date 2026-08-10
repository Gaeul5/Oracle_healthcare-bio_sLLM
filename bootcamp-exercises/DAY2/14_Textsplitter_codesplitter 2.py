from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language 

python_code = Path("./data/Sample_Code.py").read_text(encoding="utf-8")
splitter = RecursiveCharacterTextSplitter.from_language(
language=Language.PYTHON,
chunk_size=300,
chunk_overlap=50,
)
chunks = splitter.split_text(python_code)
print(f"--- CodeSplitter 결과: 총 {len(chunks)}개---")
for i, chunk in enumerate(chunks):
    print("=" * 50)
    print(f"chunk {i+1}")
    print(chunk)
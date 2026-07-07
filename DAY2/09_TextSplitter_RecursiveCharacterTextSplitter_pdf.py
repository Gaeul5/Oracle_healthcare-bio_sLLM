from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# PDF 경로 설정
spri_pdf = Path("./data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")
sample_pdf = Path("./data/Sample_RAG_Knowledge_Base.pdf")
pdf_path = spri_pdf if spri_pdf.exists() else sample_pdf
if not pdf_path.exists():
    raise FileNotFoundError("PDF 파일을 찾을 수 없습니다. ./data 폴더를 확인하세요.")
# 1. PDF 로드
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()
print(f"로드한 PDF: {pdf_path.name}")
print(f"로드된 페이지 수: {len(pages)}")
print()
# 2. chunk_size를 바꿔가며 비교
for size in [500, 1500]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.2),
        separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(pages)
print("=" * 70)
print(f"PDF: {pdf_path.name}")
print(f"chunk_size={size}")
print(f"chunk_overlap={int(size * 0.2)}")
print(f"chunk_count={len(chunks)}")
print()
# 첫 번째 chunk 정보 확인
print("[첫 번째 chunk metadata]")
print(chunks[0].metadata)
print()
print("[첫 번째 chunk preview]")
print(chunks[0].page_content[:500])
print()
# 전체 chunk 확인
print("[전체 chunk 출력]")
for i, chunk in enumerate(chunks):
    print("-" * 70)
    print(f"{i + 1}번 청크")
    print(f"길이: {len(chunk.page_content)}")
    print(f"metadata: {chunk.metadata}")
    print("내용:")
    print(repr(chunk.page_content))
    print()
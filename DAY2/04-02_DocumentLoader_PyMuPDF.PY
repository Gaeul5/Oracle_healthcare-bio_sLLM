# pip install pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
pdf_filepath = Path("./data/000660_SK_2023.pdf")

loader = PyMuPDFLoader(pdf_filepath)
pages = loader.load()

print(len(pages))
print(pages[0].page_content)
print(pages[0].metadata)
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
markdown_text = Path("./data/Sample_Markdown.md").read_text(encoding="utf-8")
headers_to_split_on = [
("#", "Header 1"),
("##", "Header 2"),
("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
print(f"--- MarkdownHeaderTextSplitter 결과: 총 {len(chunks)}개---")
for i, chunk in enumerate(chunks):
    print("=" * 50)
    print(f"chunk {i+1}")
    print("metadata:", chunk.metadata)
    print("page_content:")
    print(chunk.page_content)
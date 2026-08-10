import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

markdown_text = Path("./data/Sample_Markdown.md").read_text(encoding="utf-8")

# 1) 헤더 기준으로 먼저 섹션 나누기
heading_pattern = re.compile(r"(?m)^(#{1,6})\s+")
matches = list(heading_pattern.finditer(markdown_text))
sections = []

if not matches:
    sections = [markdown_text.strip()]
else:
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        section = markdown_text[start:end].strip()
        if section:
            sections.append(section)

print(f"헤더 기반 섹션 수: {len(sections)}")

for section_idx, section in enumerate(sections, 1):
    print("=" * 70)
    print(f"섹션 {section_idx}")
    print(section[:140].replace("\n", " ") + "...")
    print()

    # 2) 각 섹션을 다시 RecursiveCharacterTextSplitter로 문자 단위로 분할
    for size in [50, 100]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=int(size * 0.2),
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = text_splitter.split_text(section)

        print(f"chunk_size={size}, chunk_overlap={int(size * 0.2)}, chunk_count={len(chunks)}")
        for i, chunk in enumerate(chunks[:5], 1):
            print(f"  {i}번 청크, 길이={len(chunk)}")
            print(repr(chunk))
            print()


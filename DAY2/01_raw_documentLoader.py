from pathlib import Path

file_path = Path("./data/Sample_Text.txt")

with file_path.open("r", encoding="utf-8") as f:
    text = f.read()

print("--- Raw Python 결과---")
print(text)
print(type(text))

raw_doc = {
    "page_content": text,
    "metadata": {
    "source": str(file_path),
    "file_name": file_path.name,
    "file_type": file_path.suffix,
},
}
print("\n--- 직접 만든 raw_doc ---")
print(raw_doc["page_content"][:300])
print(raw_doc["metadata"])
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path="./data/Sample_CSV.csv", encoding="utf-8")
rows = loader.load()
print(f"--- CSVLoader 결과: 총 {len(rows)}개 Document ---")
for i, row in enumerate(rows):
    print("=" * 50)
    print(f"row {i}")
    print("metadata:", row.metadata)
    print("page_content:")
    print(row.page_content)
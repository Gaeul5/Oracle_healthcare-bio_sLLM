import langchain_community.document_loaders

loader = langchain_community.document_loaders.TextLoader("./data/Sample_Text.txt", encoding="utf-8")
docs = loader.load()
print("--- TextLoader 결과---")
print(f"메타데이터: {docs[0].metadata}\n")
print("--- 내용---")
print(docs[0].page_content)
print("docs type: ", type(docs))
print("doc type: ", type(docs[0]))
print("page_content: ", docs[0].page_content[:300])
print("metadata: ", docs[0].metadata)
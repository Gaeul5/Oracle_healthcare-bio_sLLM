from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

spri_pdf = Path("./data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")
sample_pdf = Path("./data/Sample_RAG_Knowledge_Base.pdf")
pdf_path = spri_pdf if spri_pdf.exists() else sample_pdf

loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

embeddings_model = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

sample_docs = documents[:3]
document_texts = [doc.page_content for doc in sample_docs]
document_embeddings = embeddings_model.embed_documents(document_texts)

print("PDF:", pdf_path.name)
print("embedding 개수:", len(document_embeddings))
print("embedding 차원:", len(document_embeddings[0]))
print("첫 번째 vector 앞 5개:", document_embeddings[0][:5])
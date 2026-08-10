"""
Day 2 과제 스타터

과제명: RAG 챗봇용 지식 베이스 준비하기

학생이 수정할 부분:
- CHUNK_SIZE를 1000에서 500으로 바꿔보고 chunk 개수 변화를 확인합니다.

목표:
- PDF가 Document로 로드되는지 확인합니다.
- Document가 chunk로 나뉘는지 확인합니다.
- 첫 번째 chunk 하나만 embedding해서 vector 길이를 확인합니다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# --------------------------------------------------
# 학생 수정 영역
# --------------------------------------------------
PDF_PATH = Path("./data/SPRi AI Brief_10월호_산업동향_1002_F.pdf")

if not PDF_PATH.exists():
    PDF_PATH = Path("./data/Sample_RAG_Knowledge_Base.pdf")

if not PDF_PATH.exists():
    raise FileNotFoundError(
        "PDF 파일을 찾을 수 없습니다. ./data 폴더에 PDF 파일이 있는지 확인하세요."
    )

CHUNK_SIZE = 500
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)


# --------------------------------------------------


# --------------------------------------------------
# 1. PDF 로드
# --------------------------------------------------
loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()


# --------------------------------------------------
# 2. Document를 chunk로 분할
# --------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# PDF에서 로드된 pages는 Document 객체 리스트입니다. 이를 splitter로 분할하여
# chunk(Document) 리스트를 생성합니다.
chunks = splitter.split_documents(pages)

if not chunks:
    raise ValueError("chunk가 생성되지 않았습니다. PDF 내용과 chunk 설정을 확인하세요.")


# --------------------------------------------------
# 3. 첫 번째 chunk 하나만 embedding
# --------------------------------------------------
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

# 첫 번째 chunk의 텍스트만 임베딩합니다.
first_vector = embeddings.embed_documents([chunks[0].page_content])[0]


# --------------------------------------------------
# 4. 결과 정리
# --------------------------------------------------
result = {
    "pdf_file": PDF_PATH.name,
    "document_count": len(pages),
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "chunk_count": len(chunks),
    "first_chunk_metadata": chunks[0].metadata,
    "first_chunk_preview": chunks[0].page_content[:500],
    "embedding_dimension": len(first_vector),
    "first_vector_preview": first_vector[:5],
}


Path("chunking_result_easy.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# --------------------------------------------------
# 5. Markdown 리포트 저장
# --------------------------------------------------
report = f"""# Day 2 RAG 챗봇용 지식 베이스 준비 결과

## 1. PDF 로드
- 파일명: {PDF_PATH.name}
- Document 개수: {len(pages)}

## 2. Chunk 결과
- CHUNK_SIZE: {CHUNK_SIZE}
- CHUNK_OVERLAP: {CHUNK_OVERLAP}
- chunk 개수: {len(chunks)}

## 3. 첫 번째 chunk metadata
```text
{chunks[0].metadata}
```

## 4. 첫 번째 chunk 미리보기
```text
{chunks[0].page_content[:500]}
```

## 5. Embedding 확인
- vector 길이: {len(first_vector)}
- 앞 5개 값: {first_vector[:5]}
"""

Path("chunking_report_easy.md").write_text(report, encoding="utf-8")
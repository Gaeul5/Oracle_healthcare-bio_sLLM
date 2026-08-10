# Day 2 RAG 챗봇용 지식 베이스 준비 결과

## 1. PDF 로드
- 파일명: SPRi AI Brief_10월호_산업동향_1002_F.pdf
- Document 개수: 29

## 2. Chunk 결과
- CHUNK_SIZE: 500
- CHUNK_OVERLAP: 100
- chunk 개수: 119

## 3. 첫 번째 chunk metadata
```text
{'producer': 'Hancom PDF 1.3.0.505', 'creator': 'Hancom PDF 1.3.0.505', 'creationdate': '2025-10-02T12:57:42+09:00', 'author': 'dj', 'moddate': '2025-10-02T12:57:42+09:00', 'pdfversion': '1.4', 'source': 'data/SPRi AI Brief_10월호_산업동향_1002_F.pdf', 'total_pages': 29, 'page': 0, 'page_label': '1'}
```

## 4. 첫 번째 chunk 미리보기
```text
2025년10월호인공지능 산업의 최신 동향
```

## 5. Embedding 확인
- vector 길이: 1536
- 앞 5개 값: [0.01311492919921875, 0.03594970703125, -0.006938934326171875, 0.0333251953125, 0.03515625]

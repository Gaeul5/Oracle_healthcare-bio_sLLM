"""Streamlit으로 가기 전, RAG 함수가 정상 동작하는지 터미널에서 한 번 확인합니다."""

from common import generate_answer, make_sources_dataframe, retrieve_documents

QUESTION = "중국 AI 플러스 정책의 핵심 영역은 무엇인가요?"
TOP_K = 4

print("Day 3 Streamlit 실습 전 RAG 함수 확인\n")
print(f"질문: {QUESTION}")
print("문서를 검색합니다...\n")

results = retrieve_documents(QUESTION, top_k=TOP_K)

print("[검색된 출처]")
source_df = make_sources_dataframe(results)
if source_df.empty:
    print("검색 결과가 없습니다.")
else:
    print(source_df.to_string(index=False))

print("\nLLM 답변을 생성합니다...\n")
answer = generate_answer(QUESTION, results)

print("=" * 80)
print("답변")
print("=" * 80)
print(answer)

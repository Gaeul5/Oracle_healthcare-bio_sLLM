import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model


# --------------------------------------------------
# 환경변수 로드
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# 설명할 개념
# --------------------------------------------------
topic = "RAG"  # TODO:딥러닝 예: "RAG"


# --------------------------------------------------
# 출력 스키마
# --------------------------------------------------
class ConceptExplanation(BaseModel):
    concept: str = Field(description="설명할 개념명")
    beginner_explanation: str = Field(description="초보자도 이해할 수 있는 쉬운 설명")
    analogy: str = Field(description="개념을 이해하기 위한 쉬운 비유")
    keywords: list[str] = Field(description="핵심 키워드 3개")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="개념 난이도")
    next_topics: list[str] = Field(description="다음에 공부하면 좋은 주제 2개")


# --------------------------------------------------
# TODO 1. PydanticOutputParser를 만드세요.
# 힌트:
# parser = PydanticOutputParser(pydantic_object=ConceptExplanation)
# --------------------------------------------------
parser = PydanticOutputParser(pydantic_object=ConceptExplanation)


# --------------------------------------------------
# TODO 2. ChatPromptTemplate을 만드세요.
# 조건:
# - system 메시지에는 AI 개념을 쉽게 설명하라는 역할을 작성합니다.
# - {format_instructions}를 반드시 포함합니다.
# - human 메시지에는 "설명할 개념: {topic}"을 넣습니다.
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 AI 개념을 쉽게 설명하는 전문가입니다.\n{format_instructions}"),
    ("human", "설명할 개념: {topic}"),
])

# --------------------------------------------------
# TODO 3. ChatOpenAI 모델을 만드세요.
# 조건:
# - model은 "gpt-4o-mini"를 사용합니다.
# - temperature는 0.2로 설정합니다.
# --------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --------------------------------------------------
# TODO 4. chain을 만드세요.
# 구조:
# chain = prompt | model | parser
# --------------------------------------------------
chain = prompt | model | parser


# --------------------------------------------------
# TODO 5. chain.invoke()를 실행해서 result를 만드세요.
# 입력값:
# - topic
# - format_instructions
# --------------------------------------------------
result = chain.invoke({
    "topic": topic,
    "format_instructions": parser.get_format_instructions(),
})


# --------------------------------------------------
# 여기부터는 제공 코드입니다. 수정하지 않아도 됩니다.
# --------------------------------------------------
results: list[ConceptExplanation] = []

results.append(result)

print("개념명:", result.concept)
print("난이도:", result.difficulty)
print("키워드:", result.keywords)
print("쉬운 설명:", result.beginner_explanation)


output_json = Path("concept_results_easy.json")

with output_json.open("w", encoding="utf-8") as f:
    json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)


report = Path("assignment_result_easy.md")

with report.open("w", encoding="utf-8") as f:
    f.write("# Day 1 쉬운 과제 결과\n\n")

    f.write("## 실행한 개념\n")
    for r in results:
        f.write(f"- {r.concept}: {r.difficulty}\n")

    f.write("\n## 확인한 점\n")
    f.write("- LLM 응답이 Pydantic 모델의 필드에 맞게 구조화되는 것을 확인했습니다.\n")
    f.write("- concept, beginner_explanation, analogy, keywords, difficulty, next_topics 필드를 확인했습니다.\n")

    f.write("\n## 소감\n")
    f.write("- 여기에 실행 후 느낀 점을 2~3줄로 작성하세요.\n")

print("\n저장 완료:", output_json, report)
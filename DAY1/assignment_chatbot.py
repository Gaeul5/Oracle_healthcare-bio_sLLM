import os

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# --------------------------------------------------
# 환경변수 로드
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 모델 & 프롬프트 (system + 대화 기록 + 이번 입력)
# --------------------------------------------------
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 한국어 챗봇입니다. 이전 대화 내용을 참고해서 답변하세요."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | model | StrOutputParser()

# --------------------------------------------------
# 대화 기록 (직접 리스트로 관리)
# --------------------------------------------------
history: list[HumanMessage | AIMessage] = []

print("챗봇을 시작합니다. 종료하려면 'exit' 또는 '종료'를 입력하세요.\n")

while True:
    user_input = input("나: ").strip()

    if user_input.lower() in ("exit", "종료"):
        print("챗봇을 종료합니다.")
        break

    if not user_input:
        continue

    print("봇: ", end="", flush=True)
    response_text = ""
    for chunk in chain.stream({"input": user_input, "history": history}):
        print(chunk, end="", flush=True)
        response_text += chunk
    print("\n")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response_text))

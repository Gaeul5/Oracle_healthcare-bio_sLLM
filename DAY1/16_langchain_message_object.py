from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
# .env 파일 로드
load_dotenv()
# 모덜 설정 및 구성
model = init_chat_model(
"openai:gpt-4.0-mini",
temperature=0.5,
timeout=300,
max_tokens=25000,
)
system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("안녕하세요? 오늘 어때요? 나와대화할 준비가 되었나요?")
# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages) # Returns AIMessage
print(response)
print(type(response))
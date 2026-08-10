from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI()
question = {"topic": "인공지능"}
prompt_text = f"{question['topic']}에 대해 쉽게 설명해줘."
response = client.chat.completions.create(
model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
messages=[
{"role": "user", "content": prompt_text}
],
)
result = response.choices[0].message.content
print(result)
print(type(result))
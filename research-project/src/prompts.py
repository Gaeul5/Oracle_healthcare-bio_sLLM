"""학생 모델 프롬프트 + 모델 ID (train_qlora.py / run_inference.py 공용).

train_qlora.py에서 분리한 이유: run_inference.py는 추론만 하면 되는데
train_qlora.py를 그대로 import하면 Trainer/TrainingArguments를 통해
datasets 패키지까지 딸려 들어와, 학습이 필요없는 추론 전용 환경에서도
datasets의 무거운(그리고 깨지기 쉬운) 의존성 트리를 요구하게 된다.
"""
from formats import TASK, RULES, PROMPTS

MODEL_IDS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

R0_PROMPT = f"""{TASK}

{RULES}
List each adverse drug event on its own line, copied exactly from the text.
If none are mentioned, output nothing.

Text:
{{text}}

Output:"""

R1_PROMPT = f"""{TASK}

{RULES}
For each adverse drug event, output one line in the form:
<exact mention from the text> -> <standard SNOMED CT term>
If no standard term applies, write "not normalized" after the arrow.
If none are mentioned, output nothing.

Text:
{{text}}

Output:"""


def build_prompt(cond, text):
    if cond == "R0":
        return R0_PROMPT.format(text=text)
    if cond == "R1":
        return R1_PROMPT.format(text=text)
    if cond in PROMPTS:
        return PROMPTS[cond].format(text=text)
    raise ValueError(f"알 수 없는 조건: {cond}")

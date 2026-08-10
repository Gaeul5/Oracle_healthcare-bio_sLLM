"""
표현 형식 R0~R4 정의 및 교사 프롬프트
=====================================

핵심 통제 (analysis_plan.md §6):
  - 교사 모델 동일 / 원문 동일 / 생성 파라미터 동일
  - 바꾸는 것은 **출력 형식뿐**
  - R2(자유서술)와 R3(구조화)의 출력 토큰 수를 ±20% 이내로 맞춘다

R2가 통제군인 이유:
  R3가 R2보다 좋다면 그것은 "구조" 때문이다.
  R2가 없으면 "교사 모델의 지식이 전이된 것"과 구분할 수 없다.
"""

# ---------------------------------------------------------------- 공통 지시문
TASK = (
    "You are given a text that may mention adverse drug events (side effects "
    "caused by a medication). Identify every adverse drug event mentioned."
)

RULES = (
    "Rules:\n"
    "- Copy each mention exactly as it appears in the text. Do not paraphrase.\n"
    "- Do not include the reason the drug was taken (indication) or a symptom "
    "that existed before the drug was started.\n"
    "- If no adverse drug event is mentioned, output nothing.\n"
)

# ------------------------------------------------------------------- R0 / R1
# 교사 모델을 쓰지 않는 조건. 원본 주석을 그대로 목표 출력으로 사용한다.

R0_SPEC = """R0 — 원문 그대로
  input : {text}
  output: 개행으로 구분된 ADE 표현 목록 (원본 주석 그대로)
"""

R1_SPEC = """R1 — 용어 정규화
  input : {text}
  output: ADE 표현 + 표준 용어 (CADEC의 MedDRA/SNOMED 매핑 사용)
          예)  extreme weight gain -> Excessive body weight gain
  ※ PHEE·PsyTAR는 UMLS/SNOMED 매핑을 사용
"""

# ------------------------------------------------------- R2 (통제군) / R3 / R4

R2_TEACHER_PROMPT = f"""{TASK}

{RULES}
Write a short paragraph (3-5 sentences) explaining which adverse drug events
are mentioned and why you identified each one. Mention the timing and the
patient's description in your explanation. Write in flowing prose, not a list.

Text:
{{text}}

Explanation:"""

R3_TEACHER_PROMPT = f"""{TASK}

{RULES}
Output one block per adverse drug event, using exactly these fields.
For onset, after_stopping, after_restarting, and severity, always answer in
one or two full sentences. If the text does not say, do not just write
"not stated" — write one or two full sentences explicitly stating that the
text does not mention it and what would need to be present for it to be
answerable (for example: "The text does not say when the nausea started. "
"There is no timestamp or day count near the mention of nausea.").

drug: <the medication named in the text>
event: <the adverse drug event, copied exactly from the text>
context: <copy the full sentence from the text that mentions this event>
onset: <a full sentence on when the event started after taking the drug>
after_stopping: <a full sentence on what happened when the drug was stopped>
after_restarting: <a full sentence on what happened when the drug was taken again>
severity: <a full sentence stating whether the event was mild, moderate, or severe, and why>

Text:
{{text}}

Output:"""

R4_TEACHER_PROMPT = f"""{TASK}

{RULES}
Output one block per adverse drug event, using exactly these fields.
For onset, after_stopping, after_restarting, and severity, always answer in
one or two full sentences. If the text does not say, do not just write
"not stated" — write one or two full sentences explicitly stating that the
text does not mention it and what would need to be present for it to be
answerable (for example: "The text does not say when the nausea started. "
"There is no timestamp or day count near the mention of nausea."). judgement
is always exactly one sentence.

drug: <the medication named in the text>
event: <the adverse drug event, copied exactly from the text>
context: <copy the full sentence from the text that mentions this event>
onset: <a full sentence on when the event started after taking the drug>
after_stopping: <a full sentence on what happened when the drug was stopped>
after_restarting: <a full sentence on what happened when the drug was taken again>
severity: <a full sentence stating whether the event was mild, moderate, or severe, and why>
judgement: <one sentence on how clearly the drug caused this event>

Text:
{{text}}

Output:"""

PROMPTS = {"R2": R2_TEACHER_PROMPT, "R3": R3_TEACHER_PROMPT, "R4": R4_TEACHER_PROMPT}

# --------------------------------------------------------------- 검증용 유틸

def token_len(s, tok=None):
    """조건 간 출력 길이 정렬 확인용. tokenizer가 없으면 공백 기준 근사."""
    if tok is None:
        return len(s.split())
    return len(tok.encode(s))


def check_length_balance(outputs_by_cond, tol=0.20):
    """R2/R3/R4 출력 토큰 수가 ±tol 이내인지 검사.

    이 검사를 통과하지 못하면 '구조 때문'이라는 주장이 성립하지 않는다.
    (더 긴 출력으로 더 많이 배웠을 가능성을 배제할 수 없기 때문)
    """
    means = {c: sum(map(token_len, v)) / len(v) for c, v in outputs_by_cond.items()}
    base = means.get("R2")
    report = {}
    for c, m in means.items():
        ratio = m / base if base else float("nan")
        report[c] = {"mean_tokens": round(m, 1),
                     "ratio_to_R2": round(ratio, 3),
                     "ok": abs(ratio - 1) <= tol}
    return report

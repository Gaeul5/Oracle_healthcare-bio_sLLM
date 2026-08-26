"""
모델 출력 -> ADE 표현 리스트 파싱
=================================
조건별 파싱 규칙:
  R0/R1 : 줄바꿈으로 구분된 목록. 각 줄이 그대로 하나의 ADE 표현
          (R1은 " -> " 앞부분만 span으로 취급).
  R3/R4 : "event: <span>" 필드를 정규식으로 추출.
  R2    : 자유서술이라 정해진 필드가 없다. 원문(source_text) 중 모델 출력에
          문자 그대로(대소문자만 무시) 등장하는 부분문자열만 후보로 채택한다
          (왼쪽에서부터 탐욕적으로 가장 긴 매치 우선, 겹치지 않게, 최대 8단어).
          모델이 완전히 다른 말로 바꿔 쓰면 recall이 낮게 나올 수 있는데,
          이는 파싱 노이즈가 아니라 "구조화되지 않은 자유서술에서 정확한
          문자열을 복원하지 못하는 실제 손실"로 해석한다 — R2 vs R3/R4
          비교(H1)의 일부다.
"""
import re

_R34_FIELD_RE = re.compile(r"^event:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_WORD_RE = re.compile(r"\S+")


def parse_r0(output_text):
    return [line.strip() for line in output_text.splitlines() if line.strip()]


def parse_r1(output_text):
    spans = []
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        span = line.split("->", 1)[0].strip()
        if span:
            spans.append(span)
    return spans


def parse_r34(output_text):
    return [m.group(1).strip() for m in _R34_FIELD_RE.finditer(output_text)]


def parse_r2(output_text, source_text, drug_name=None, max_ngram=8):
    """원문 부분문자열 매칭으로 R2 자유서술에서 ADE 후보를 추출한다."""
    if not source_text:
        return []
    norm_output = re.sub(r"\s+", " ", output_text.lower())
    tokens = list(_WORD_RE.finditer(source_text))
    n = len(tokens)
    drug_lower = (drug_name or "").lower().strip()
    spans = []
    i = 0
    while i < n:
        matched = None
        for length in range(min(max_ngram, n - i), 0, -1):
            start = tokens[i].start()
            end = tokens[i + length - 1].end()
            candidate = source_text[start:end]
            norm_candidate = re.sub(r"\s+", " ", candidate.lower())
            if norm_candidate and norm_candidate in norm_output and norm_candidate != drug_lower:
                matched = candidate
                i += length
                break
        if matched:
            spans.append(matched)
        else:
            i += 1
    return spans


def parse_prediction(condition, output_text, source_text=None, drug_name=None):
    if condition == "R0":
        return parse_r0(output_text)
    if condition == "R1":
        return parse_r1(output_text)
    if condition in ("R3", "R4"):
        return parse_r34(output_text)
    if condition == "R2":
        return parse_r2(output_text, source_text, drug_name)
    raise ValueError(f"알 수 없는 조건: {condition}")


if __name__ == "__main__":
    # 자체 테스트 (합성 예시) — GPU 없이 로직만 검증
    r0_out = "extreme weight gain\nhair loss\n"
    assert parse_prediction("R0", r0_out) == ["extreme weight gain", "hair loss"]

    r1_out = "extreme weight gain -> Excessive body weight gain\nhair loss -> not normalized\n"
    assert parse_prediction("R1", r1_out) == ["extreme weight gain", "hair loss"]

    r34_out = (
        "drug: Losartan\nevent: bp up and down\ncontext: bp up and down but\n"
        "onset: The text does not say.\nafter_stopping: The text does not say.\n"
        "after_restarting: The text does not say.\nseverity: Mild.\n\n"
        "drug: Losartan\nevent: headaches\ncontext: headaches after going off.\n"
        "onset: unclear.\nafter_stopping: unclear.\nafter_restarting: unclear.\n"
        "severity: moderate.\njudgement: Likely caused by the drug.\n"
    )
    assert parse_prediction("R3", r34_out) == ["bp up and down", "headaches"]
    assert parse_prediction("R4", r34_out) == ["bp up and down", "headaches"]

    source = "3 days no problems as of yet.... bp up and down but 1 good reading 124/76"
    r2_out = (
        "The patient reports that their blood pressure went bp up and down "
        "after starting the medication, which suggests an adverse reaction."
    )
    preds = parse_prediction("R2", r2_out, source_text=source, drug_name="Cozaar")
    assert "bp up and down" in preds, preds

    print("모든 자체 테스트 통과")

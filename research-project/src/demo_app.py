"""
발표용 라이브 데모 — 조건(R0/R2/R3/R4)별 ADE 추출 비교
========================================================
학교 GPU 세션에서 직접 실행. 같은 입력 문장에 대해 R0/R2/R3/R4로 학습된
어댑터의 실시간 출력을 나란히 보여준다. gradio의 share=True로 공개 URL을
띄우므로 Backend.AI 포트 설정과 무관하게 브라우저에서 바로 접속 가능하다.

실행:
  python3 demo_app.py
"""
import os
import re
import sys

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import MODEL_IDS, build_prompt  # noqa: E402
from parse_predictions import parse_r0, parse_r2  # noqa: E402

_FIELD_RE = re.compile(r"^(drug|event):\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def summarize(cond, output_text, source_text=None, drug_name=None):
    """모델이 뱉는 원시 추출 결과를 사람이 읽기 쉬운 한 줄 머리말로 바꾼다."""
    if cond in ("R3", "R4"):
        pairs, cur_drug = [], None
        for m in _FIELD_RE.finditer(output_text):
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "drug":
                cur_drug = val
            elif key == "event":
                pairs.append((cur_drug or "약물 미상", val))
        if not pairs:
            return "이상반응 언급 없음"
        return "\n".join(f"🔎 이상반응 감지됨: {d} → {e}" for d, e in pairs)
    if cond == "R0":
        spans = parse_r0(output_text)
        return f"🔎 감지된 이상반응: {', '.join(spans)}" if spans else "이상반응 언급 없음"
    if cond == "R2":
        # 연구용 채점(parse_predictions.parse_r2)은 그대로 두고, 데모 화면에
        # 보여줄 요약에서만 너무 짧은 매치(예: "I", "a")를 걸러 잡음을 줄인다.
        spans = [s for s in parse_r2(output_text, source_text or "", drug_name) if len(s) >= 3]
        return f"🔎 감지된 이상반응(추정): {', '.join(spans)}" if spans else "이상반응 언급 없음"
    return ""

ADAPTERS_DIR = "/home/work/oracle/adapters"
CONDITIONS = ["R0", "R2", "R3", "R4"]
MAX_NEW_TOKENS = {"R0": 80, "R2": 180, "R3": 220, "R4": 240}
SEED = 0

EXAMPLE_TEXT = (
    "I've been on Losartan for about two weeks now. Started getting bad "
    "headaches and my blood pressure readings have been bouncing up and "
    "down, which is confusing since that's what the medication is supposed "
    "to help with."
)

_cache = {}
_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_combo(model_size, train_domain):
    key = (model_size, train_domain)
    if key in _cache:
        return _cache[key]

    # 새 조합을 얹기 전에 이전 조합을 GPU에서 완전히 비운다. 이 clear를
    # 새 모델 로드 "뒤"에 하면 old+new가 동시에 GPU에 상주하는 순간이
    # 생겨 OOM으로 데모 프로세스가 통째로 죽는다(실제로 한 번 죽었음).
    for old_tok, old_model in _cache.values():
        del old_model
    _cache.clear()
    torch.cuda.empty_cache()

    model_id = MODEL_IDS[model_size]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=_bnb, device_map="auto")

    model = None
    for cond in CONDITIONS:
        adapter_dir = f"{ADAPTERS_DIR}/{model_size}_{cond}_{train_domain}_seed{SEED}"
        if model is None:
            model = PeftModel.from_pretrained(base, adapter_dir, adapter_name=cond)
        else:
            model.load_adapter(adapter_dir, adapter_name=cond)
    model.eval()

    _cache[key] = (tok, model)
    return tok, model


def generate_one(tok, model, cond, text):
    model.set_adapter(cond)
    prompt = build_prompt(cond, text)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        gen = model.generate(
            ids, attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS[cond],
            min_new_tokens=3,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()


def run(model_size, train_domain, text):
    if not text.strip():
        return ["(입력 문장을 넣어주세요)"] * 4
    tok, model = load_combo(model_size, train_domain)
    outputs = []
    for cond in CONDITIONS:
        raw = generate_one(tok, model, cond, text)
        summary = summarize(cond, raw, source_text=text)
        outputs.append(f"{summary}\n{'─' * 24}\n{raw}")
    return outputs


with gr.Blocks(title="ADE 추출 형식 비교 데모") as demo:
    gr.Markdown(
        "# ADE 추출 — 표현 형식(R0/R2/R3/R4) 실시간 비교\n"
        "같은 문장을 4가지 형식으로 학습된 어댑터에 동시에 넣어 출력을 비교합니다. "
        "R0은 원문 그대로, R2는 교사 자유서술 통제군, R3/R4는 구조화 형식입니다."
    )
    with gr.Row():
        model_size = gr.Dropdown(["0.5b", "1.5b", "3b"], value="3b", label="모델 크기")
        train_domain = gr.Dropdown(["forum", "literature"], value="forum", label="학습 도메인 (미학습 도메인 텍스트를 넣어보세요)")
    text = gr.Textbox(value=EXAMPLE_TEXT, lines=4, label="입력 문장")
    btn = gr.Button("생성", variant="primary")
    with gr.Row():
        out_r0 = gr.Textbox(label="R0 — 원문 그대로", lines=6)
        out_r2 = gr.Textbox(label="R2 — 교사 자유서술 (통제군)", lines=6)
        out_r3 = gr.Textbox(label="R3 — 구조화 필드", lines=6)
        out_r4 = gr.Textbox(label="R4 — 구조화 + 판단 근거", lines=6)

    btn.click(run, inputs=[model_size, train_domain, text], outputs=[out_r0, out_r2, out_r3, out_r4])

demo.queue().launch(share=True, server_name="0.0.0.0")

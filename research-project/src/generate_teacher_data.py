"""
R2/R3/R4 교사 데이터 생성 (Qwen2.5-7B-Instruct, 4bit)
====================================================
Colab T4 세션에서 실행. 학생 모델 학습과 같은 세션에서 돌리지 말 것
(CLAUDE.md §5 — VRAM 부족 위험).

analysis_plan.md §6 통제 조건:
  - 교사 모델 / 원문 / 생성 파라미터가 R2·R3·R4 사이에서 동일해야 하며
    바뀌는 것은 프롬프트(출력 형식)뿐이다. GEN_KWARGS를 세 조건이 공유한다.
  - 오픈웨이트 모델만 사용 (Qwen2.5-7B-Instruct).

체크포인트: --out 파일에 이미 있는 doc_id는 건너뛰고 이어서 생성한다.
문서 1건이 끝날 때마다 한 줄씩 기록하므로, 세션이 중간에 끊겨도
잘린 레코드가 남지 않는다.

Colab 사용 예:
  from google.colab import drive; drive.mount('/content/drive')
  !pip install -q -U transformers accelerate bitsandbytes
  # 먼저 소량으로 속도/VRAM 확인 (전체 실행 전 필수)
  !python generate_teacher_data.py \\
      --unified /content/drive/MyDrive/ade-project/unified.jsonl \\
      --out /content/drive/MyDrive/ade-project/teacher_outputs.jsonl \\
      --limit 20
  # 확인 후 --limit 없이 전체 실행 (여러 세션에 걸쳐 이어서 실행 가능)
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formats import PROMPTS, check_length_balance  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CONDITIONS = ("R2", "R3", "R4")

# R2/R3/R4가 공유하는 생성 파라미터. greedy decoding으로 고정해
# 조건 간 비교에서 샘플링 변동성을 배제한다.
GEN_KWARGS = dict(
    max_new_tokens=350,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.1,
)


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
    )
    model.eval()
    return tok, model


def load_done_ids(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


@torch.inference_mode()
def generate_one(tok, model, prompt_template, text):
    prompt = prompt_template.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids, pad_token_id=tok.pad_token_id, **GEN_KWARGS)
    new_tokens = out[0, input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=20,
                     help="진행 상황을 출력하는 간격(문서 수). 저장 자체는 매 문서마다 수행")
    ap.add_argument("--limit", type=int, default=None,
                     help="디버그/속도 확인용. 생략 시 전체 문서 처리")
    ap.add_argument("--max-input-tokens", type=int, default=1024,
                     help="입력 원문 길이 상한. 99.5%%ile이 약 400토큰이라 대부분 잘리지 않음")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_json(args.unified, lines=True)
    if args.limit:
        df = df.head(args.limit)

    done = load_done_ids(args.out)
    todo = df[~df.doc_id.isin(done)]
    print(f"전체 {len(df)}건 / 이미 완료 {len(done)}건 / 남은 {len(todo)}건")

    if len(todo) == 0:
        print("모두 완료됨.")
        return

    tok, model = load_model()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            ids = tok(row.text, truncation=True,
                      max_length=args.max_input_tokens)["input_ids"]
            text_trunc = tok.decode(ids, skip_special_tokens=True)

            record = {"doc_id": row.doc_id}
            for cond in CONDITIONS:
                record[cond] = generate_one(tok, model, PROMPTS[cond], text_trunc)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if i % args.checkpoint_every == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed
                remain_min = (len(todo) - i) / rate / 60 if rate > 0 else float("nan")
                print(f"[{i}/{len(todo)}] {elapsed:.0f}s 경과, "
                      f"남은 예상 {remain_min:.1f}분")

    print("생성 완료.")

    all_records = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
    outputs_by_cond = {c: [r[c] for r in all_records] for c in CONDITIONS}
    report = check_length_balance(outputs_by_cond)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    bad = [c for c, v in report.items() if not v["ok"]]
    if bad:
        print(f"경고: {bad} 조건이 R2 대비 ±20% 범위를 벗어남. "
              f"프롬프트 길이 조정 필요 여부를 검토할 것.")


if __name__ == "__main__":
    main()

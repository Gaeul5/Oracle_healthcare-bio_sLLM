"""
학생 모델(어댑터) 추론 — 평가셋에 대한 예측 생성
================================================
train_qlora.py로 학습된 LoRA 어댑터 하나를 로드해, 그 학습 도메인의
반대쪽 도메인(미학습 도메인)에 대해 고정 평가셋(make_eval_sample.py로
생성)에서 예측을 생성한다. 프롬프트는 train_qlora.py의 build_prompt를
그대로 재사용해 학습 때와 동일한 형식을 보장한다.

체크포인트: --out 파일에 이미 있는 doc_id는 건너뛰고 이어서 생성한다
(Colab/원격 인스턴스 세션이 끊겨도 처음부터 다시 돌릴 필요 없음).

사용 예:
  python run_inference.py \\
      --unified /workspace/data/unified.jsonl \\
      --eval-sample research-project/data/eval_sample.csv \\
      --adapter-dir /workspace/adapters/0.5b_R3_forum_seed0 \\
      --out /workspace/adapters/0.5b_R3_forum_seed0/predictions.jsonl
"""
import argparse
import csv
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import MODEL_IDS, build_prompt  # noqa: E402
from run_naming import RUN_NAME_RE, OPPOSITE  # noqa: E402

MAX_NEW_TOKENS = {"R0": 80, "R1": 100, "R2": 180, "R3": 220, "R4": 240}


def parse_run_name(adapter_dir):
    name = os.path.basename(os.path.normpath(adapter_dir))
    m = RUN_NAME_RE.match(name)
    if not m:
        raise ValueError(
            f"어댑터 디렉토리 이름에서 조건을 못 읽음: {name} "
            f"(--model-size/--condition/--train-domain을 직접 지정할 것)"
        )
    return m.group("size"), m.group("cond"), m.group("domain")


def load_eval_docs(unified_path, eval_sample_path, eval_domain):
    wanted = set()
    with open(eval_sample_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["domain"] == eval_domain:
                wanted.add(row["doc_id"])

    docs = []
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["doc_id"] in wanted:
                docs.append(d)
    if len(docs) != len(wanted):
        print(f"경고: 평가 샘플 {len(wanted)}건 중 unified.jsonl에서 {len(docs)}건만 찾음")
    return docs


def load_done_ids(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["doc_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--eval-sample", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-size", choices=MODEL_IDS.keys(), default=None)
    ap.add_argument("--condition", choices=["R0", "R1", "R2", "R3", "R4"], default=None)
    ap.add_argument("--train-domain", choices=["forum", "literature"], default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    parsed_size, parsed_cond, parsed_domain = parse_run_name(args.adapter_dir)
    model_size = args.model_size or parsed_size
    condition = args.condition or parsed_cond
    train_domain = args.train_domain or parsed_domain
    eval_domain = OPPOSITE[train_domain]
    out_path = args.out or os.path.join(args.adapter_dir, "predictions.jsonl")
    max_new = args.max_new_tokens or MAX_NEW_TOKENS[condition]

    print(f"어댑터={args.adapter_dir} 모델={model_size} 조건={condition} "
          f"학습도메인={train_domain} -> 평가도메인={eval_domain}")

    docs = load_eval_docs(args.unified, args.eval_sample, eval_domain)
    done = load_done_ids(out_path)
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"평가 문서 {len(docs)}건, 이미 완료 {len(done)}건, 남음 {len(todo)}건")

    if not todo:
        print("남은 문서 없음, 종료")
        return

    model_id = MODEL_IDS[model_size]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fout:
        n_done = 0
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            prompts = [build_prompt(condition, row["text"]) for row in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    # Qwen2.5-Instruct의 기본 eos_token(<|im_end|>)에 대한 강한 사전학습
                    # prior가 LoRA로 학습한 실제 콘텐츠 확률을 근소하게 앞서, 첫 토큰에서
                    # 바로 EOS로 collapse하는 경우가 관측됨(빈 출력 다발). min_new_tokens로
                    # 최소 몇 토큰은 강제 생성시켜 우회한다.
                    min_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            completions = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for row, completion in zip(batch, completions):
                fout.write(json.dumps({"doc_id": row["doc_id"], "output_text": completion}, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += len(batch)
            print(f"[{n_done}/{len(todo)}] batch of {len(batch)} done")

    print(f"완료. 예측 저장: {out_path}")


if __name__ == "__main__":
    main()

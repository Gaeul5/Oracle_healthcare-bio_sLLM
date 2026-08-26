"""
RAG(검색+few-shot) vs QLoRA 비교 — 파인튜닝 없이 검색만으로 R3 형식 흉내
========================================================================
72개 조합 그리드와 별개의 보조 실험. QLoRA로 가중치를 갱신하는 대신,
학습 도메인 문서 + 그 R3 교사 출력을 검색 풀로 삼아 BM25로 유사 문서
k개를 찾아 few-shot 예시로 프롬프트에 넣고, 파인튜닝하지 않은 베이스
모델로 같은 R3 형식 출력을 생성한다. 같은 고정 평가셋(400건)과 같은
채점 파이프라인(parse_predictions + evaluate.py)을 그대로 재사용해
기존 3B/R3 QLoRA 결과와 직접 비교 가능하게 한다.

시드가 없는 이유: BM25 검색은 결정론적이라 반복 실행해도 같은 k개가
나온다 — 학습 조건별로 3번 반복하던 QLoRA와 달리 방향당 1회만 필요.

실행 예:
  python run_rag_eval.py --unified data/unified.jsonl \
      --teacher-outputs data/teacher_outputs.jsonl \
      --eval-sample data/eval_sample.csv \
      --train-domain forum --model-size 3b --k 3 \
      --out reports/rag_forum_literature.jsonl
"""
import argparse
import csv
import json
import os
import re
import sys

import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import MODEL_IDS  # noqa: E402
from formats import TASK, RULES  # noqa: E402
from run_naming import OPPOSITE  # noqa: E402
from parse_predictions import parse_r34  # noqa: E402
from evaluate import evaluate  # noqa: E402

FIELD_INSTRUCTIONS = (
    "Output one block per adverse drug event, using exactly these fields.\n"
    "For onset, after_stopping, after_restarting, and severity, always answer in\n"
    "exactly one full sentence. If the text does not say, do not just write\n"
    '"not stated" — write one full sentence explicitly stating that the text\n'
    "does not mention it.\n\n"
    "drug: <the medication named in the text>\n"
    "event: <the adverse drug event, copied exactly from the text>\n"
    "context: <copy the full sentence from the text that mentions this event>\n"
    "onset: <a full sentence on when the event started after taking the drug>\n"
    "after_stopping: <a full sentence on what happened when the drug was stopped>\n"
    "after_restarting: <a full sentence on what happened when the drug was taken again>\n"
    "severity: <a full sentence stating whether the event was mild, moderate, or severe, and why>\n"
)
PREAMBLE = f"{TASK}\n\n{RULES}\n{FIELD_INSTRUCTIONS}"

_TOKEN_RE = re.compile(r"\w+")


def tokenize(s):
    return _TOKEN_RE.findall(s.lower())


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(demos, target_text):
    parts = [PREAMBLE]
    for demo_text, demo_output in demos:
        parts.append(f"Text:\n{demo_text}\n\nOutput:\n{demo_output.strip()}")
    parts.append(f"Text:\n{target_text}\n\nOutput:")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--teacher-outputs", required=True)
    ap.add_argument("--eval-sample", required=True)
    ap.add_argument("--train-domain", choices=["forum", "literature"], required=True)
    ap.add_argument("--model-size", choices=MODEL_IDS.keys(), default="3b")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_domain = OPPOSITE[args.train_domain]
    unified = load_jsonl(args.unified)
    teacher_by_id = {r["doc_id"]: r for r in load_jsonl(args.teacher_outputs)}

    # 검색 풀: 학습 도메인 문서 중 R3 교사 출력이 있는 것만
    pool = [r for r in unified if r["domain"] == args.train_domain and r["doc_id"] in teacher_by_id]
    pool_tokens = [tokenize(r["text"]) for r in pool]
    bm25 = BM25Okapi(pool_tokens)
    print(f"검색 풀: {len(pool)}건 ({args.train_domain})")

    wanted_ids = set()
    with open(args.eval_sample, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["domain"] == eval_domain:
                wanted_ids.add(row["doc_id"])
    eval_docs = [r for r in unified if r["doc_id"] in wanted_ids]
    print(f"평가 문서: {len(eval_docs)}건 ({eval_domain})")

    model_id = MODEL_IDS[args.model_size]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pairs = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for start in range(0, len(eval_docs), args.batch_size):
            batch = eval_docs[start:start + args.batch_size]
            prompts = []
            for row in batch:
                scores = bm25.get_scores(tokenize(row["text"]))
                top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:args.k]
                demos = [(pool[i]["text"], teacher_by_id[pool[i]["doc_id"]]["R3"]) for i in top_idx]
                prompts.append(build_prompt(demos, row["text"]))

            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=220, min_new_tokens=3,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            completions = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

            for row, completion in zip(batch, completions):
                fout.write(json.dumps({"doc_id": row["doc_id"], "output_text": completion}, ensure_ascii=False) + "\n")
                preds = parse_r34(completion)
                pairs.append((preds, row["ade_spans"]))
            fout.flush()
            print(f"[{min(start + args.batch_size, len(eval_docs))}/{len(eval_docs)}] 완료")

    result = evaluate(pairs)
    result["n_docs"] = len(pairs)
    print(f"\n{args.train_domain} -> {eval_domain} (RAG, k={args.k}, {args.model_size}):")
    print(f"  strict  P={result['strict']['precision']} R={result['strict']['recall']} F1={result['strict']['f1']}")
    print(f"  relaxed P={result['relaxed']['precision']} R={result['relaxed']['recall']} F1={result['relaxed']['f1']}")
    with open(args.out + ".summary.json", "w", encoding="utf-8") as f:
        json.dump({"train_domain": args.train_domain, "eval_domain": eval_domain,
                    "model_size": args.model_size, "k": args.k, **result}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

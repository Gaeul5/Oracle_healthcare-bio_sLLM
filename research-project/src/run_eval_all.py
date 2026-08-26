"""
전체 어댑터 결과 집계
=====================
run_inference.py가 각 어댑터 디렉토리에 남긴 predictions.jsonl을 읽어,
조건별 파싱(parse_predictions.py) + strict/relaxed F1(evaluate.py)을
계산하고 조합별 결과표를 CSV로 저장한다. GPU 불필요, predictions.jsonl이
있는 조합만 채점하고 없는 조합은 목록으로 알려준다(추론이 아직 안 끝난
어댑터를 나중에 다시 확인할 때 씀).
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_naming import RUN_NAME_RE, OPPOSITE  # noqa: E402
from parse_predictions import parse_prediction  # noqa: E402
from evaluate import evaluate  # noqa: E402


def load_unified_by_id(unified_path, wanted_ids):
    by_id = {}
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["doc_id"] in wanted_ids:
                by_id[d["doc_id"]] = d
    return by_id


def score_run(run_dir, run_name, condition, eval_domain, unified_path):
    pred_path = os.path.join(run_dir, "predictions.jsonl")
    if not os.path.exists(pred_path):
        return None

    preds_by_id = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds_by_id[r["doc_id"]] = r["output_text"]

    gold_by_id = load_unified_by_id(unified_path, set(preds_by_id))

    pairs = []
    n_missing_gold = 0
    for doc_id, output_text in preds_by_id.items():
        row = gold_by_id.get(doc_id)
        if row is None:
            n_missing_gold += 1
            continue
        preds = parse_prediction(
            condition, output_text,
            source_text=row["text"], drug_name=row.get("drug_name"),
        )
        pairs.append((preds, row["ade_spans"]))

    if n_missing_gold:
        print(f"  경고: {run_name} — gold 못 찾은 doc {n_missing_gold}건 제외")
    if not pairs:
        return None

    result = evaluate(pairs)
    result["n_docs"] = len(pairs)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--adapters-dir", required=True)
    ap.add_argument("--out", default="research-project/reports/eval_results.csv")
    args = ap.parse_args()

    run_dirs = sorted(
        d for d in os.listdir(args.adapters_dir)
        if RUN_NAME_RE.match(d) and os.path.isdir(os.path.join(args.adapters_dir, d))
    )

    rows = []
    missing = []
    for run_name in run_dirs:
        m = RUN_NAME_RE.match(run_name)
        model_size, condition, train_domain, seed = (
            m.group("size"), m.group("cond"), m.group("domain"), m.group("seed"),
        )
        eval_domain = OPPOSITE[train_domain]
        run_dir = os.path.join(args.adapters_dir, run_name)

        result = score_run(run_dir, run_name, condition, eval_domain, args.unified)
        if result is None:
            missing.append(run_name)
            continue

        rows.append({
            "run_name": run_name,
            "model_size": model_size,
            "condition": condition,
            "train_domain": train_domain,
            "eval_domain": eval_domain,
            "seed": seed,
            "n_docs": result["n_docs"],
            "strict_p": result["strict"]["precision"],
            "strict_r": result["strict"]["recall"],
            "strict_f1": result["strict"]["f1"],
            "relaxed_p": result["relaxed"]["precision"],
            "relaxed_r": result["relaxed"]["recall"],
            "relaxed_f1": result["relaxed"]["f1"],
        })
        print(f"{run_name}: strict F1={result['strict']['f1']} "
              f"relaxed F1={result['relaxed']['f1']} (n={result['n_docs']})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)}개 조합 결과 저장: {args.out}")

    if missing:
        print(f"\n예측 없음(추론 미완료) {len(missing)}개: {missing}")


if __name__ == "__main__":
    main()

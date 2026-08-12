"""
R2/R3/R4 교사 데이터 생성 — vLLM 배치 추론 버전 (실험적, T4 처리량 미검증)
================================================================
generate_teacher_data.py(HF transformers 순차 생성, 문서당 실측 ~32.8초)의
처리량 문제를 해결하기 위한 vLLM 기반 버전. continuous batching으로
문서당 시간을 크게 줄이는 게 목표이나, T4(Turing, compute capability 7.5)는
vLLM이 최적화를 가정하는 Ampere 이상 GPU가 아니라서 실제 개선폭은
스모크 테스트로 직접 확인해야 한다. HF 버전은 generate_teacher_data_hf_backup.py로
그대로 보존되어 있으니, 여기서 문제가 생기면 그쪽으로 되돌아갈 것.

교사 모델은 동일 베이스(Qwen2.5-7B-Instruct)의 AWQ/GPTQ 사전 양자화 배포판을
쓴다 — bitsandbytes NF4는 vLLM에서 안정적으로 지원되지 않기 때문.
CLAUDE.md §5에 "GPTQ 또는 bnb-4bit" 둘 다 허용 옵션으로 명시되어 있어
통제 조건(analysis_plan.md §6 "교사 모델 동일") 위반은 아니라고 판단함
(같은 베이스 모델의 다른 양자화 배포판).

생성 파라미터는 generate_teacher_data.py의 GEN_KWARGS와 동일하게 맞춘다
(temperature=0 == do_sample=False인 greedy decoding, repetition_penalty=1.1,
max_tokens=350 == max_new_tokens=350).

go/no-go 기준 (Claude Code 세션 기록):
  1. 설치 + 모델 로딩이 1시간 안에 성공
  2. --limit 20 스모크 테스트 문서당 처리 시간이 HF 대비 2배 이상 빠름
     (HF 실측 32.8초/건 기준 → 16.4초/건 이하)
  둘 다 만족해야 go. 하나라도 실패하면 HF 버전으로 복귀.

Colab 사용 예:
  !pip install -q -U vllm
  # 먼저 20건 스모크 테스트로 처리량 확인 (go/no-go 판단용, 필수)
  !python src/generate_teacher_data_vllm.py \\
      --unified /content/drive/MyDrive/ade-project/unified.jsonl \\
      --out /content/drive/MyDrive/ade-project/teacher_outputs_vllm_test.jsonl \\
      --limit 20
  # AWQ 로딩이 실패하면 GPTQ 체크포인트로 재시도:
  #   --model-id Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formats import PROMPTS, check_length_balance  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
CONDITIONS = ("R2", "R3", "R4")

# generate_teacher_data.py의 GEN_KWARGS와 동일한 생성 파라미터.
SAMPLING_KWARGS = dict(
    max_tokens=350,
    temperature=0,
    repetition_penalty=1.1,
)


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


def build_prompt(tok, prompt_template, text):
    prompt = prompt_template.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default=MODEL_ID,
                     help="vLLM으로 로드할 사전 양자화 체크포인트")
    ap.add_argument("--quantization", default="awq", choices=["awq", "gptq"],
                     help="--model-id를 GPTQ 체크포인트로 바꾸면 gptq로 지정")
    ap.add_argument("--batch-docs", type=int, default=20,
                     help="vLLM 배치 1회에 묶을 문서 수 (문서당 프롬프트 3개씩 배치됨)")
    ap.add_argument("--limit", type=int, default=None,
                     help="디버그/속도 확인용. 생략 시 전체 문서 처리")
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    args = ap.parse_args()

    if not os.path.exists(args.unified):
        raise SystemExit(f"--unified 경로에 파일이 없음: {args.unified}")

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_json(args.unified, lines=True)
    if args.limit:
        df = df.head(args.limit)

    done = load_done_ids(args.out)
    todo = df[~df.doc_id.isin(done)]
    print(f"전체 {len(df)}건 / 이미 완료 {len(done)}건 / 남은 {len(todo)}건")
    if len(todo) == 0:
        print("모두 완료됨.")
        return

    tok = AutoTokenizer.from_pretrained(args.model_id)
    llm = LLM(model=args.model_id, quantization=args.quantization,
              max_model_len=3072, gpu_memory_utilization=0.90, dtype="float16")
    sampling_params = SamplingParams(**SAMPLING_KWARGS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    t0 = time.time()
    n_done = 0
    rows = list(todo.itertuples())
    with open(args.out, "a", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_docs):
            chunk = rows[start:start + args.batch_docs]

            prompts = []
            meta = []  # (doc_id, cond), 프롬프트와 같은 순서로 대응
            for row in chunk:
                ids = tok(row.text, truncation=True,
                          max_length=args.max_input_tokens)["input_ids"]
                text_trunc = tok.decode(ids, skip_special_tokens=True)
                for cond in CONDITIONS:
                    prompts.append(build_prompt(tok, PROMPTS[cond], text_trunc))
                    meta.append((row.doc_id, cond))

            # vLLM은 입력 순서 그대로 출력을 반환한다 (공식 문서 기준).
            outputs = llm.generate(prompts, sampling_params)

            records = {}
            for (doc_id, cond), out in zip(meta, outputs):
                records.setdefault(doc_id, {"doc_id": doc_id})
                records[doc_id][cond] = out.outputs[0].text.strip()

            for row in chunk:
                f.write(json.dumps(records[row.doc_id], ensure_ascii=False) + "\n")
            f.flush()

            n_done += len(chunk)
            elapsed = time.time() - t0
            per_doc = elapsed / n_done
            remain_min = (len(rows) - n_done) * per_doc / 60
            print(f"[{n_done}/{len(rows)}] {elapsed:.0f}s 경과, "
                  f"문서당 {per_doc:.1f}s (HF 실측 32.8s 대비 {32.8/per_doc:.1f}배), "
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
        print(f"경고: {bad} 조건이 R2 대비 ±20% 범위를 벗어남.")


if __name__ == "__main__":
    main()

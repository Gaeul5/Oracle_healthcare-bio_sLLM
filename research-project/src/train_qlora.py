"""
학생 모델 QLoRA 학습
====================
Colab T4 세션에서 실행. 교사 데이터 생성(generate_teacher_data.py)과
같은 세션에서 돌리지 말 것 (CLAUDE.md §5 — VRAM 부족 위험).

지원 조건: R0, R1, R2, R3, R4
  R1(용어 정규화)은 CADEC v1의 SNOMED CT 매핑(normalize_cadec_v1.py의 출력,
  --cadec-v1-sct)만 사용 가능하다. CADECv2/PsyTAR/PHEE는 매핑이 없어
  --condition R1일 때 자동으로 corpus=="cadec_v1" 문서로만 좁혀지며,
  몇 건이 제외됐는지 로그로 출력한다. --train-domain literature와
  R1을 같이 쓰면(리터러처에는 매핑이 전혀 없음) 즉시 에러를 낸다.

두 가지 학습 축 (analysis_plan.md §9):
  --train-domain {forum,literature}
      주 축. 도메인 전체로 학습, 반대쪽 도메인에서 평가(평가는 별도 스크립트).
  --cv-round {1,2,3,4}
      보조 축. 포럼 내부 약물 그룹 4-fold 중 해당 회차의 train 역할 문서만
      사용 (data/fold_assignment.csv의 cv_round{n} 컬럼). 폴드 간 vs 시드 간
      표준편차 비교(우선순위 5번)를 위한 저비용 실험용.
  둘 중 정확히 하나를 지정해야 한다.

학습 입력 프롬프트는 R2/R3/R4의 경우 교사에게 준 것과 동일한 프롬프트
(formats.PROMPTS)를 그대로 재사용한다 — 형식(출력 구조)만 학습 목표가
바뀌고, 과제 지시문은 통제 조건과 일관되게 유지하기 위함.

Colab 사용 예:
  !pip install -q -U transformers accelerate bitsandbytes peft datasets
  !python train_qlora.py \\
      --unified /content/drive/MyDrive/ade-project/unified.jsonl \\
      --teacher-outputs /content/drive/MyDrive/ade-project/teacher_outputs.jsonl \\
      --model-size 0.5b --condition R3 --train-domain forum \\
      --seed 0 --out-dir /content/drive/MyDrive/ade-project/adapters/0.5b_R3_forum_seed0

  # R1 (CADEC v1 SNOMED CT만, cv_round 1 또는 4에서만 학습 예시가 생김)
  !python train_qlora.py \\
      --unified /content/drive/MyDrive/ade-project/unified.jsonl \\
      --cadec-v1-sct /content/drive/MyDrive/ade-project/cadec_v1_sct.jsonl \\
      --fold-assignment /content/drive/MyDrive/ade-project/fold_assignment.csv \\
      --model-size 0.5b --condition R1 --cv-round 1 \\
      --seed 0 --out-dir /content/drive/MyDrive/ade-project/adapters/0.5b_R1_round1_seed0
"""
import argparse
import json
import os
import sys

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import MODEL_IDS, build_prompt  # noqa: E402

# T4 16GB 기준 모델 크기별 기본 배치 설정. 필요시 CLI로 덮어쓴다.
# gradient checkpointing 적용 후에도 Qwen2.5의 큰 vocab(~152k)이 fp32 loss
# 계산에서 logits 텐서를 크게 만들어 OOM 위험이 있어, per-device 배치를
# 작게 유지하고 grad accumulation으로 유효 배치(=16)를 모델 크기 간 통일한다.
BATCH_DEFAULTS = {
    "0.5b": dict(per_device_train_batch_size=4, gradient_accumulation_steps=4),
    "1.5b": dict(per_device_train_batch_size=2, gradient_accumulation_steps=8),
    "3b": dict(per_device_train_batch_size=1, gradient_accumulation_steps=16),
}

def build_target(cond, row, teacher_by_doc, sct_by_doc=None):
    if cond == "R0":
        return "\n".join(row["ade_spans"])
    if cond == "R1":
        terms = sct_by_doc[row["doc_id"]]["ade_terms"]
        lines = [f"{span} -> {term or 'not normalized'}"
                 for span, term in zip(row["ade_spans"], terms)]
        return "\n".join(lines)
    return teacher_by_doc[row["doc_id"]][cond]


def load_examples(args):
    import pandas as pd
    df = pd.read_json(args.unified, lines=True)

    if args.condition == "R1" and args.train_domain == "literature":
        raise ValueError("R1은 literature 도메인에 매핑 데이터가 전혀 없음 "
                          "(CADEC v1 SNOMED CT만 지원)")

    if args.train_domain:
        df = df[df.domain == args.train_domain]
    else:
        fold = pd.read_csv(args.fold_assignment)
        col = f"cv_round{args.cv_round}"
        train_ids = set(fold.loc[fold[col] == "train", "doc_id"])
        df = df[df.doc_id.isin(train_ids)]

    teacher_by_doc, sct_by_doc = {}, None
    if args.condition == "R1":
        if not args.cadec_v1_sct:
            raise ValueError("조건 R1에는 --cadec-v1-sct 필요 (normalize_cadec_v1.py 출력)")
        sct_by_doc = {}
        with open(args.cadec_v1_sct, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sct_by_doc[r["doc_id"]] = r
        before = len(df)
        df = df[df.corpus == "cadec_v1"]
        excluded = before - len(df)
        if excluded:
            print(f"경고: R1은 CADEC v1 SNOMED CT 매핑만 지원 — "
                  f"corpus!=cadec_v1 문서 {excluded}건 제외 (전체 {before}건 중 {len(df)}건 사용)")
        if len(df) == 0:
            raise ValueError(
                "R1 학습 예시가 0건. CADEC v1의 두 약물 그룹(STATIN, NSAID)이 "
                "모두 fold_assignment.csv에서 fold=2에 속해 있어, cv_round 2·3에서는 "
                "cadec_v1 문서가 전부 calib/test 역할이 되고 train에 하나도 남지 않음. "
                "R1은 --cv-round 1 또는 4만 사용할 것."
            )
    elif args.condition != "R0":
        if not args.teacher_outputs:
            raise ValueError(f"조건 {args.condition}에는 --teacher-outputs 필요")
        with open(args.teacher_outputs, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    teacher_by_doc[r["doc_id"]] = r
        before = len(df)
        df = df[df.doc_id.isin(teacher_by_doc)]
        missing = before - len(df)
        if missing:
            print(f"경고: 교사 데이터 없는 문서 {missing}건 제외 (전체 {before}건 중)")

    examples = []
    for _, row in df.iterrows():
        prompt = build_prompt(args.condition, row["text"])
        target = build_target(args.condition, row, teacher_by_doc, sct_by_doc)
        examples.append({"prompt": prompt, "target": target})
    return examples


def make_tokenize_fn(tok, max_len):
    def fn(ex):
        prompt_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tok(ex["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}
    return fn


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--teacher-outputs", default=None,
                     help="R2/R3/R4에 필요. generate_teacher_data.py의 출력")
    ap.add_argument("--cadec-v1-sct", default=None,
                     help="R1에 필요. normalize_cadec_v1.py의 출력")
    ap.add_argument("--fold-assignment", default=None,
                     help="--cv-round 사용 시 필요")
    ap.add_argument("--model-size", choices=MODEL_IDS.keys(), required=True)
    ap.add_argument("--condition", choices=["R0", "R1", "R2", "R3", "R4"], required=True)
    ap.add_argument("--train-domain", choices=["forum", "literature"], default=None)
    ap.add_argument("--cv-round", type=int, choices=[1, 2, 3, 4], default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=None,
                     help="생략 시 모델 크기별 기본값 사용")
    ap.add_argument("--grad-accum", type=int, default=None)
    args = ap.parse_args()

    if bool(args.train_domain) == bool(args.cv_round):
        raise ValueError("--train-domain과 --cv-round 중 정확히 하나만 지정할 것")

    torch.manual_seed(args.seed)

    examples = load_examples(args)
    print(f"학습 예시 {len(examples)}건 "
          f"(조건={args.condition}, "
          f"{'도메인=' + args.train_domain if args.train_domain else 'cv_round=' + str(args.cv_round)})")

    model_id = MODEL_IDS[args.model_size]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    ds = Dataset.from_list(examples).map(
        make_tokenize_fn(tok, args.max_len), remove_columns=["prompt", "target"]
    )

    batch_defaults = BATCH_DEFAULTS[args.model_size]
    per_device_bs = args.batch_size or batch_defaults["per_device_train_batch_size"]
    grad_accum = args.grad_accum or batch_defaults["gradient_accumulation_steps"]

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"어댑터 저장: {args.out_dir}")


if __name__ == "__main__":
    main()

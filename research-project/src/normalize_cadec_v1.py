"""
CADEC v1 SNOMED CT 정규화 (R1 조건용)
=====================================
CADEC v1(Karimi 2015) 원본에는 `sct/*.ann`에 ADR 스팬별 SNOMED CT 코드·
정식 용어가 이미 들어있어(외부 사전 조회 불필요), R1을 이 코퍼스에 한해
바로 만들 수 있다.

CADECv2(Dai 2024)/PsyTAR/PHEE는 이런 매핑이 없고, PsyTAR·PHEE는 UMLS
정규화가 별도로 필요하다 (analysis_plan.md 참고, 이 스크립트의 범위 밖).

매칭 방식
---------
`original/*.ann`의 T<n>과 `sct/*.ann`의 TT<n>은 같은 엔티티를 가리키며
(같은 숫자 접미사), 오프셋이 아니라 이 ID로 조인해야 안전하다 — CADEC v1
ADR의 15.9%가 불연속 스팬(fragment가 여러 개)인데, sct 파일은 그 fragment
구조를 그대로 보존하기 때문에 순서가 다르면 오프셋 문자열 비교가 깨질 수
있다. `build_unified.py`의 `parse_brat_merged`와 동일한 방식(fragment
정렬 후 최소 포함 구간으로 병합, 병합 결과가 겹치면 먼저 나온 것만 유지)
으로 재구현해 ID를 함께 추적한다 — 이래야 여기서 만든 ade_offsets이
unified.jsonl과 정확히 일치한다 (스크립트 끝에서 직접 대조 검증한다).

SNOMED 코드가 없는 경우(`CONCEPT_LESS`) 또는 sct 파일에 해당 ID 자체가
없는 경우 term은 None으로 남긴다 (전체 ADR의 약 3.8%, 3.9.md `--out`
리포트에 수치로 남는다).

출력: data/cadec_v1_sct.jsonl
  {"doc_id": ..., "ade_offsets": [[s,e], ...], "ade_terms": [term|None, ...]}
  ade_offsets/ade_terms는 같은 순서로 정렬되어 unified.jsonl의 해당 문서
  ade_offsets과 zip 가능하다.
"""
import argparse
import json
import os
import re
import sys

BASE_DEFAULT = os.environ.get("ADE_DATA_ROOT", "/home/claude/work")


def _parse_adr_with_ids(ann_path):
    """build_unified.parse_brat_merged와 동일한 병합·중복 제거 로직이되
    살아남은 엔티티의 원본 T-id를 함께 반환한다."""
    if not os.path.exists(ann_path):
        return []
    seen, out = set(), []
    for line in open(ann_path, encoding="utf-8", errors="replace"):
        if not line.startswith("T") or line.startswith("TT"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        tid = parts[0]
        bits = parts[1].split(" ", 1)
        if len(bits) < 2 or bits[0] != "ADR":
            continue
        frags = []
        for seg in bits[1].split(";"):
            try:
                a, b = seg.split()
            except ValueError:
                continue
            frags.append((int(a), int(b)))
        if not frags:
            continue
        frags.sort()
        span = (frags[0][0], frags[-1][1])
        if span in seen:
            continue
        seen.add(span)
        out.append((span, tid))
    out.sort(key=lambda x: x[0])
    return out


_TERM_RE = re.compile(r"\|\s*([^|]+?)\s*\|")


def _parse_sct_terms(ann_path):
    """TT<n> -> 정규화 용어(문자열) 또는 None (CONCEPT_LESS / 코드 없음)."""
    terms = {}
    if not os.path.exists(ann_path):
        return terms
    for line in open(ann_path, encoding="utf-8", errors="replace"):
        if not line.startswith("TT"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        tid = "T" + parts[0][2:]  # "TT12" -> "T12" (original id와 동일)
        val = parts[1]
        if val.startswith("CONCEPT_LESS"):
            terms[tid] = None
            continue
        found = _TERM_RE.findall(val)  # "code | term |" 형태에서 term만 추출
        terms[tid] = "; ".join(dict.fromkeys(found)) if found else None
    return terms


def process_corpus(base, unified_path):
    import pandas as pd
    df = pd.read_json(unified_path, lines=True)
    df = df[df.corpus == "cadec_v1"]
    keep_stems = {row.doc_id.split(":", 1)[1]: row for _, row in df.iterrows()}

    root = f"{base}/cadec/v2/cadec"
    rows = []
    n_total = n_mapped = 0
    mismatches = []

    for stem, urow in keep_stems.items():
        orig_path = f"{root}/original/{stem}.ann"
        sct_path = f"{root}/sct/{stem}.ann"
        entities = _parse_adr_with_ids(orig_path)
        term_of = _parse_sct_terms(sct_path)

        offsets = [list(span) for span, _ in entities]
        terms = [term_of.get(tid) for _, tid in entities]
        n_total += len(terms)
        n_mapped += sum(1 for t in terms if t)

        if offsets != [list(o) for o in urow["ade_offsets"]]:
            mismatches.append(stem)

        rows.append({"doc_id": urow["doc_id"], "ade_offsets": offsets,
                     "ade_terms": terms})

    return rows, n_total, n_mapped, mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--base", default=BASE_DEFAULT,
                     help="ADE_DATA_ROOT와 동일한 원본 데이터 루트 (build_unified.py 참고)")
    ap.add_argument("--out", default=None,
                     help="생략 시 unified.jsonl과 같은 폴더의 cadec_v1_sct.jsonl")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.unified)), "cadec_v1_sct.jsonl")

    rows, n_total, n_mapped, mismatches = process_corpus(args.base, args.unified)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"문서 {len(rows)}건, ADR 스팬 {n_total}개 중 {n_mapped}개 "
          f"({n_mapped / n_total:.1%}) SNOMED CT 용어 매핑됨")
    print(f"저장: {out_path}")

    if mismatches:
        print(f"\n경고: unified.jsonl의 ade_offsets과 어긋난 문서 {len(mismatches)}건: "
              f"{mismatches[:10]}{' ...' if len(mismatches) > 10 else ''}")
        print("build_unified.py의 병합 로직과 이 스크립트가 어긋난 것이므로 "
              "결과를 그대로 쓰면 안 됨 — 원인 확인 필요.")
    else:
        print("검증 OK: 모든 문서에서 ade_offsets이 unified.jsonl과 정확히 일치.")


if __name__ == "__main__":
    main()

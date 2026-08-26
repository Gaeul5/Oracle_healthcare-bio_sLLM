"""
평가셋 고정 샘플링
==================
주 축(forum <-> literature) 평가는 상대 도메인 전체(문헌 4,824건 / 포럼
5,516건)를 쓰기엔 72개 어댑터 x 전체 문서 조합이 계산상 불가능하다
(약 37만 건 생성 필요). 대신 도메인당 400건을 한 번만 무작위로 뽑아
고정하고, 같은 도메인으로 학습된 모든 조합(3 크기 x 4 조건 x 3 시드 = 24개)이
동일한 평가셋을 공유한다. analysis_plan.md §14의 "평가셋 200건 미만" 하한의
2배로, 마감(발표 3일 전) 안에 72개 조합 추론을 끝내기 위한 축소다.
doc_id만 저장하므로(원문 없음) fold_assignment.csv와 같은 방식으로 git에
커밋 가능하다.
"""
import argparse
import csv
import json
import random
from collections import defaultdict

N_PER_DOMAIN = 400
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True)
    ap.add_argument("--out", default="research-project/data/eval_sample.csv")
    args = ap.parse_args()

    by_domain = defaultdict(list)
    with open(args.unified, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                by_domain[d["domain"]].append(d["doc_id"])

    rng = random.Random(SEED)
    rows = []
    for domain, doc_ids in sorted(by_domain.items()):
        n = min(N_PER_DOMAIN, len(doc_ids))
        sample = rng.sample(doc_ids, n)
        rows.extend((doc_id, domain) for doc_id in sample)
        print(f"{domain}: 전체 {len(doc_ids)}건 중 {n}건 샘플링")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "domain"])
        writer.writerows(rows)
    print(f"저장: {args.out} ({len(rows)}행)")


if __name__ == "__main__":
    main()

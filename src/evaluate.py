"""
ADE 추출 평가
=============

MultiADE 논문은 엄격(strict) 매칭과 완화(relaxed) 매칭을 모두 언급하지만,
논문에 명시된 GitHub 저장소(github.com/daixiangau/MultiADE)는 2026-08 기준
접근할 수 없다(Not Found). 따라서 평가 기준을 여기에 직접 정의하고,
본 연구의 수치는 논문 수치와 **직접 비교하지 않고** 경향 비교만 한다.
이 사실을 논문에 명시한다.

정의
----
strict  : 문자열이 정규화 후 완전히 일치
relaxed : 정규화 후 한쪽이 다른 쪽을 포함하거나 토큰 자카드 >= 0.5

정규화: 소문자화 / 구두점 제거 / 공백 정리 / 관사·소유격 제거
"""
import re
from collections import Counter

_STOP = {"a", "an", "the", "my", "his", "her", "their", "our", "your", "some"}


def normalise(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks)


def _jaccard(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if A | B else 0.0


def match(pred, gold, mode="strict"):
    if mode == "strict":
        return pred == gold
    if pred == gold:
        return True
    if pred and gold and (pred in gold or gold in pred):
        return True
    return _jaccard(pred, gold) >= 0.5


def score_doc(preds, golds, mode="strict"):
    """한 문서의 TP/FP/FN. 중복 표현은 multiset으로 취급."""
    P = [normalise(p) for p in preds if str(p).strip()]
    G = [normalise(g) for g in golds if str(g).strip()]
    used, tp = [False] * len(G), 0
    for p in P:
        for i, g in enumerate(G):
            if not used[i] and match(p, g, mode):
                used[i] = True
                tp += 1
                break
    return tp, len(P) - tp, len(G) - tp


def prf(pairs, mode="strict"):
    """pairs = [(preds, golds), ...] -> micro P/R/F1"""
    TP = FP = FN = 0
    for preds, golds in pairs:
        a, b, c = score_doc(preds, golds, mode)
        TP += a; FP += b; FN += c
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1": round(f * 100, 2),
            "tp": TP, "fp": FP, "fn": FN}


def evaluate(pairs):
    """엄격·완화 두 기준을 항상 함께 보고한다."""
    return {"strict": prf(pairs, "strict"), "relaxed": prf(pairs, "relaxed")}


# ------------------------------------------------------------ 선택적 예측
def risk_coverage(pairs_with_conf, mode="relaxed", steps=20):
    """확신도 순으로 정렬해 하위부터 기권시키며 (coverage, F1)을 반환.

    pairs_with_conf = [(preds, golds, confidence), ...]
    """
    items = sorted(pairs_with_conf, key=lambda x: -x[2])
    n, curve = len(items), []
    for k in range(1, steps + 1):
        m = max(1, round(n * k / steps))
        sub = [(p, g) for p, g, _ in items[:m]]
        curve.append({"coverage": round(m / n, 3), **prf(sub, mode)})
    return curve


if __name__ == "__main__":
    demo = [(["extreme weight gain", "hair loss"],
             ["extreme weight gain", "hair loss", "memory loss"]),
            (["my nausea"], ["nausea"])]
    import json
    print(json.dumps(evaluate(demo), indent=2, ensure_ascii=False))

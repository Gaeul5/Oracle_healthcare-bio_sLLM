"""
실험 분할 고정
==============
축 1 (주): 도메인 이전  forum <-> literature
축 2 (보조): 포럼 내부 약물 그룹 4-폴드 교차검증

한 번 생성한 뒤에는 절대 변경하지 않는다 (analysis_plan.md §9).
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

# 그룹 크기를 최대한 균형 있게 4폴드로 묶는다 (greedy bin packing).
# 결과를 코드에 고정해 재현성을 보장한다.
FOLD_OF_GROUP = {
    "ANTIDEPRESSANT": 0,        # 1796
    "ACID_RELATED": 1,          # 1113
    "STATIN": 2,                # 997
    "ANTIDEPRESSANT_PSY": 3,    # 835
    "ANTIHYPERTENSIVE": 3,      # 414  -> 폴드3 = 1249
    "NSAID": 2,                 # 361  -> 폴드2 = 1358
}

# 4폴드 회전: 각 회차마다 평가 1 / 보정 1 / 학습 2
ROTATION = [
    # (평가 폴드, 보정 폴드)
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]


def main():
    df = pd.read_json(os.path.join(DATA, "unified.jsonl"), lines=True)

    # ---- 축 1: 도메인 이전 ----
    df["axis1_role"] = df.domain.map({"forum": "forum", "literature": "literature"})

    # ---- 축 2: 포럼 내부 그룹 폴드 ----
    df["group_fold"] = df.drug_group.map(FOLD_OF_GROUP).fillna(-1).astype(int)

    for i, (te, cal) in enumerate(ROTATION):
        col = f"cv_round{i+1}"
        def role(f):
            if f < 0:
                return "na"          # literature
            if f == te:
                return "test"
            if f == cal:
                return "calib"
            return "train"
        df[col] = df.group_fold.map(role)

    cols = ["doc_id", "domain", "corpus", "drug_group", "group_fold",
            "axis1_role"] + [f"cv_round{i+1}" for i in range(4)]
    out = df[cols]
    path = os.path.join(DATA, "fold_assignment.csv")
    out.to_csv(path, index=False)

    print(f"저장: {path}  ({len(out)}건)\n")
    print("=== 축 1: 도메인 이전 ===")
    print(df.axis1_role.value_counts().to_string(), "\n")
    print("=== 축 2: 포럼 그룹 폴드 ===")
    print(df[df.domain == "forum"].groupby(["group_fold", "drug_group"]).size().to_string(), "\n")
    print("=== 회차별 문서 수 (포럼) ===")
    f = df[df.domain == "forum"]
    print(pd.DataFrame({f"round{i+1}": f[f"cv_round{i+1}"].value_counts()
                        for i in range(4)}).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()

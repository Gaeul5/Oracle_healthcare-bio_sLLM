# Oracle_healthcare-bio_sLLM

Healthcare & Bio sLLM 과정 학습 자료와, 이를 바탕으로 진행하는 개인 연구 프로젝트를 함께 관리하는 저장소.

## 디렉터리 구조

```
.
├── research-project/     # 개인 연구: 표현 형식이 sLLM ADE 추출 일반화에 미치는 영향
└── bootcamp-exercises/   # 과정 실습 및 과제 (RAG, LangGraph, MCP 등)
```

---

## research-project

# 환자 서사 표현 형식과 소형 언어모델의 도메인 일반화

학습 데이터의 **표현 형식**이 소형 생성 언어모델(0.5B–7B)의
**도메인 간 이상반응(ADE) 추출 일반화**에 미치는 영향을 통제 실험으로 측정한다.

연구 질문·가설·평가 지표·중단 기준은 **실험 시작 전에**
[`research-project/analysis_plan.md`](research-project/analysis_plan.md)에 고정되어 있다.

### ⚠️ 데이터는 이 저장소에 포함되지 않는다

원본 코퍼스는 각자의 라이선스에 따라 **직접 내려받아야 한다.**
특히 CADEC v1은 CSIRO Data Licence 하에 있어 재배포가 허용되지 않는다.

| 코퍼스 | 확보 경로 | 라이선스 |
|---|---|---|
| CADECv2 | CSIRO Data Portal `csiro:62387` (DOI 10.25919/3v5b-k950) | CC BY |
| CADEC v1 | CSIRO Data Portal (DOI 10.4225/08/570FB102BDAD2) | CSIRO Data Licence (연구 목적) |
| PsyTAR | Data in Brief 논문 Online Supplement #1 / AskaPatient 연구 페이지 | CC BY 4.0 |
| PHEE | 원 저장소 | 원 저장소 기준 |

n2c2와 MADE 1.0은 본 연구에서 사용하지 않는다. 사유는 `analysis_plan.md` §7 참조.

### 디렉터리 배치 (데이터)

```
<ADE_DATA_ROOT>/
├── cadecv2_new/data/extracted/data/{txt,ann,split}/
├── cadec/v2/cadec/{text,original}/
├── psytar/psytar/PsyTAR_dataset.xlsx
└── PHEE-master/PHEE-master/data/json/{train,dev,test}.json
```

### 실행 파이프라인

```bash
cd research-project
pip install pandas openpyxl
export ADE_DATA_ROOT=/path/to/corpora
python src/build_unified.py     # -> data/unified.jsonl (gitignore 대상)
```

이후 단계 (GPU 필요, 별도 세션 권장):

```bash
# 1. 교사 모델(Qwen2.5-7B-Instruct, 4bit)로 R2/R3/R4 생성
python src/generate_teacher_data.py --unified data/unified.jsonl --out data/teacher_outputs.jsonl

# 2. 고정 평가셋 샘플링 (도메인당 400건, 1회만 실행)
python src/make_eval_sample.py --unified data/unified.jsonl --out data/eval_sample.csv

# 3. 학생 모델(Qwen2.5 0.5B/1.5B/3B) QLoRA 학습 — 조건 x 도메인 x 시드 조합별 반복
python src/train_qlora.py --unified data/unified.jsonl --teacher-outputs data/teacher_outputs.jsonl \
    --model-size 0.5b --condition R3 --train-domain forum --seed 0 \
    --out-dir adapters/0.5b_R3_forum_seed0

# 4. 미학습 도메인에 대해 추론 후 채점 (어댑터 디렉토리당 1회)
python src/run_inference.py --unified data/unified.jsonl --eval-sample data/eval_sample.csv \
    --adapter-dir adapters/0.5b_R3_forum_seed0
python src/run_eval_all.py --unified data/unified.jsonl --adapters-dir adapters \
    --out reports/eval_results.csv
```

학습된 어댑터(`adapters/`)는 용량이 커서(72개 조합 기준 약 1.4GB) 이 저장소에는 포함하지 않는다.

### 결과 요약

72개 조합(모델 3종 × 조건 4종 × 도메인 2종 × 시드 3개)을 전부 학습·평가했다.
전체 수치는 [`research-project/reports/eval_results.csv`](research-project/reports/eval_results.csv)에 있다.

- **H1 지지** — 구조화 형식(R3/R4)이 자유서술 통제군(R2)보다 strict F1 기준으로 18/18 조합에서
  전부 높았다.
- **H2 기각(반대 방향)** — 구조화의 효과 크기(R3−R2)는 모델이 작을수록 커질 것으로 예상했으나,
  실제로는 모델이 클수록 커졌다 (forum 학습 기준 0.5B +39.5pt → 1.5B +44.2pt → 3B +48.9pt).
- R2가 R0보다도 낮게 나오는 것은 자유서술에서 정확한 문자열을 복원하는 채점 파싱의 한계가
  섞인 결과이며, 논문 Limitations에 명시한다.

발표 자료: [`research-project/reports/presentation.pptx`](research-project/reports/presentation.pptx) ·
[`presentation_script.md`](research-project/reports/presentation_script.md).
조건별(R0/R2/R3/R4) 출력을 실시간으로 비교하는 데모는 `research-project/src/demo_app.py`로 실행한다
(GPU 세션에서 `python src/demo_app.py` — gradio 공개 링크 생성).

### 통합 스키마

| 필드 | 설명 |
|---|---|
| `doc_id` | 전역 고유 ID (`corpus:stem`) |
| `domain` | `forum` \| `literature` |
| `corpus` | `cadecv2` \| `cadec_v1` \| `psytar` \| `phee` |
| `drug_group` | 약물 계열 (그룹 교차검증 키) |
| `drug_name` | 약물명 |
| `text` | 원문 |
| `ade_spans` | ADE 표현 목록 |
| `ade_offsets` | `(start, end)` 목록 — 불연속은 병합됨 |
| `n_discontig_merged` | 병합한 불연속 엔티티 수 |
| `verbatim_ratio` | 표현이 원문에 그대로 등장하는 비율 |
| `orig_split` | 원본 데이터셋의 공식 분할 |

집계 통계는 [`research-project/reports/data_stats.md`](research-project/reports/data_stats.md)에 있다 (원문 미포함).

### 참고 문헌

- Dai X., Karimi S., Sarker A., Hachey B., Paris C. **MultiADE: A Multi-domain
  Benchmark for Adverse Drug Event Extraction.** *J Biomed Inform* 160:104744, 2024.
- Karimi S., Metke-Jimenez A., Kemp M., Wang C. **CADEC: A corpus of adverse drug
  event annotations.** *J Biomed Inform* 55:73–81, 2015.
- Zolnoori M., et al. **The PsyTAR dataset.** *Data in Brief* 24:103838, 2019.

---

## bootcamp-exercises

Healthcare & Bio sLLM 과정 진행 중 작성한 실습 코드와 과제 모음 (RAG 챗봇, LangGraph, MCP, A2A 등).
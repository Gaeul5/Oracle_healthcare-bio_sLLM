# 프로젝트 컨텍스트 (Claude Code 세션 시작용)

이 파일은 Claude(웹 채팅)와의 이전 세션들에서 이어지는 연구 프로젝트의 맥락을 담고 있다.
새 세션을 시작하는 Claude Code는 작업 전에 이 문서 전체와 `research-project/analysis_plan.md`를 먼저 읽을 것.

---

## 1. 이 저장소의 구조

GitHub: `Gaeul5/Oracle_healthcare-bio_sLLM` (origin, main 브랜치). VSCode/Colab 등
어느 환경에서 세션을 열든 git clone으로 최신 코드를 받을 수 있다.

```
.
├── research-project/     # 아래에서 설명하는 연구 프로젝트 (현재 작업 대상)
│   ├── analysis_plan.md  # 사전등록 문서 — 최우선으로 읽을 것
│   ├── data/
│   │   ├── fold_assignment.csv   # 폴드 배정 (git에 있음, 원문 없어 커밋 가능)
│   │   ├── unified.jsonl         # 통합 코퍼스 (원문 포함 → gitignore, 로컬에서만 존재)
│   │   ├── cadec_v1_sct.jsonl    # CADEC v1 R1용 SNOMED CT 정규화 (원문 포함 → gitignore)
│   │   └── raw/                  # 원본 zip 압축 해제본 + ADE_DATA_ROOT 심볼릭 링크 구조
│   │                              # (gitignore, build_unified.py 실행에 필요)
│   ├── reports/
│   │   └── data_stats.md   # 파생 통계만 (원문 없음, 커밋됨)
│   ├── notebooks/          # Colab 러너 노트북 (git clone + !python으로 src/*.py 실행)
│   │   ├── generate_teacher_data.ipynb
│   │   └── train_qlora.ipynb
│   └── src/
│       ├── build_unified.py       # CADEC/CADECv2/PsyTAR/PHEE → 통합 스키마 JSONL (완성)
│       ├── formats.py             # R0~R4 표현 형식 정의 + R2/R3/R4 교사 프롬프트 (완성)
│       ├── make_folds.py          # 폴드 배정 스크립트 (완성, 결과 고정됨)
│       ├── evaluate.py            # strict/relaxed F1 평가 모듈 (완성, 아직 실전 연결 안 됨)
│       ├── normalize_cadec_v1.py  # CADEC v1 SNOMED CT 파싱 → R1 타겟 (완성)
│       ├── generate_teacher_data.py  # 교사(Qwen2.5-7B 4bit)로 R2/R3/R4 생성 (완성, 미실행)
│       └── train_qlora.py         # 학생 모델 QLoRA 학습, R0/R1/R2/R3/R4 지원 (완성, 미실행)
└── bootcamp-exercises/    # 별개 과정 실습 자료. 이 연구와 무관, 건드릴 필요 없음
```

## 2. 연구 질문 (확정됨, 변경 금지 — 사전등록 완료)

학습 데이터의 **표현 형식(representation format)**이 소형 생성 언어모델(0.5B–3B)의
**도메인 간 ADE(약물 이상반응) 추출 일반화**에 영향을 주는가.
그 효과는 모델 크기가 작을수록 커지는가.

- **H1**: 인과 구조를 명시한 형식(R3/R4)이, 동일 교사·동일 원문의 자유서술 형식(R2)보다
  미학습 도메인에서 높은 ADE 추출 F1을 보인다.
- **H2**: H1의 효과 크기는 모델이 작을수록 크다.
- H1/H2가 기각되어도 결과로 보고한다 (falsifiable, 사전등록됨).

일반화 축: **PHEE(문헌) ↔ CADEC+PsyTAR(포럼)** 양방향 도메인 전이.
MultiADE(Dai et al. 2024) 벤치마크가 보고한 참고 기준선(F1):
PHEE→PsyTAR 6.5, PHEE→CADEC 13.1 — 문헌→포럼 방향이 특히 어렵다는 비대칭 확인됨.

## 3. 실험 조건 (R0~R4) — `formats.py`에 프롬프트까지 완성됨

| ID | 형식 | 상태 |
|---|---|---|
| R0 | 원문 그대로 | 완료. `train_qlora.py`에서 바로 학습 가능 |
| R1 | 용어 정규화 (SNOMED CT) | **CADEC v1만** 완료 (`normalize_cadec_v1.py`, 매핑률 96.2%). CADECv2/PsyTAR/PHEE는 UMLS 정규화 필요 — UTS 계정 신청함, NLM 승인 대기 중(§4 참고). MedDRA는 별도 유료 라이선스 필요해 처음부터 배제, SNOMED CT만 사용 |
| R2 | 교사 자유서술 — **통제군** | 프롬프트 완성, 교사 모델 확정(`Qwen2.5-7B-Instruct` 4bit). `generate_teacher_data.py`로 생성 가능하나 **아직 Colab에서 실행한 적 없음** — `teacher_outputs.jsonl` 존재하지 않음 |
| R3 | 구조화 필드 | R2와 동일 상태 |
| R4 | 구조화 + 판단 근거 한 줄 | R2와 동일 상태 |

R1은 CADEC v1(1,155건, forum 5,516건 중 21%)로만 학습 가능하고 forum 내부
4-fold CV 중 round 1·4에서만 학습 예시가 생긴다 (round 2·3은 cadec_v1의 두
약물 그룹이 전부 fold 2라 train 예시 0건 — `train_qlora.py`가 이 경우 명시적
에러를 던진다). literature 도메인에는 R1 데이터가 전혀 없어 주 축(forum↔literature)
실험에는 CADECv2/PsyTAR/PHEE 정규화가 끝나기 전까지 못 쓴다.

**절대 위반 금지 통제 조건** (`formats.py`의 `check_length_balance()`로 검증):
- 교사 모델 동일, 원문 동일, 생성 파라미터 동일
- R2/R3/R4 출력 토큰 수 ±20% 이내로 정렬
- 교사는 **오픈웨이트 모델만** 사용 (상용 API 출력으로 학습 금지 — 라이선스/재현성 문제)

## 4. 데이터 (모두 1차 채널로 직접 확보 완료, 검증 완료)

| 코퍼스 | 도메인 | 문서 수 | 라이선스 | 확보 경로 |
|---|---|---|---|---|
| CADECv2 | forum | 3,526 | CC BY | CSIRO Data Portal (csiro:62387) |
| CADEC v1 | forum | 1,155 | CSIRO Data Licence | CSIRO Data Portal (DOI 10.4225/08/570FB102BDAD2) |
| PsyTAR | forum | 835 | CC BY 4.0 | Data in Brief 논문 Online Supplement |
| PHEE | literature | 4,824 | 원 저장소 기준 | 원 GitHub 저장소 |

이미 해결된 데이터 이슈 (다시 검토할 필요 없음):
- PsyTAR annotation이 잘못된 텍스트 필드를 참조해 span 80% 유실 → 수정 완료
- CADEC v1/v2 간 104개 중복 문서 식별됨 → 통합 시 처리 필요
- CADEC(2016)와 CADECv2(MultiADE 확장, 2024)는 별개 데이터셋 — 혼동 주의
- n2c2, MADE 1.0: 개인 연구자가 접근 불가(IRB/팀 필요)로 명시적으로 제외
- SMM4H: 트위터 재현성 문제로 제외
- `build_unified.py`의 전역 중복 제거가 비안정 정렬(quicksort)을 써서 동률
  처리가 실행마다 달라지던 재현성 버그 발견·수정함 (`kind="stable"`). 2회
  실행 동일 해시로 검증됨. 이미 고쳐졌으니 재확인 불필요.

**UMLS 상태**: PsyTAR/PHEE R1 정규화에 필요한 UTS 계정을 본인 명의로 신청함
(신청일 2026-08-10, NLM 승인까지 최대 3영업일). 승인 이메일이 오기 전까지는
이 작업을 진행할 수 없다 — 승인되면 사용자에게 물어보고 정규화 매칭
스크립트(SNOMED CT API 또는 QuickUMLS)를 새로 작성해야 한다. MedDRA는
UMLS 라이선스로도 못 얻는다(MSSO 별도 구독 필요) — SNOMED CT만 목표로 한다.

## 5. 컴퓨팅 환경 (확정)

**Google Colab, T4 GPU (16GB VRAM)가 사용 가능한 최대 사양.** 로컬 GPU 없음.
이 제약을 벗어나는 방법(더 큰 모델, 풀 파인튜닝 등)은 제안하지 말 것.

- 교사 모델(R2/R3/R4 생성용, 학습 없이 추론만): `Qwen2.5-7B-Instruct`, 4bit 양자화
  (GPTQ 또는 bnb-4bit). T4에서 추론은 가능하나 여유 VRAM이 크지 않으므로 배치 크기에 주의.
- 학생 모델(실험 대상, 실제 파인튜닝 대상): `Qwen2.5-0.5B-Instruct` / `Qwen2.5-1.5B-Instruct`
  / `Qwen2.5-3B-Instruct` — 교사와 같은 Qwen2.5 계열로 통일 (아키텍처 차이가 아니라
  순수 크기 효과만 비교하기 위함, H2 검증에 중요).
- 학습 방법: QLoRA (4bit 양자화 + LoRA 어댑터). T4에서 7B 풀 파인튜닝은 불가능하지만
  0.5B~3B QLoRA는 가능함. 교사 모델은 학습시키지 않고 추론에만 사용.
- **세션 분리 필수**: 교사 모델 추론(R2/R3/R4 생성)과 학생 모델 학습을 같은 Colab
  세션에서 동시에 돌리면 VRAM 부족 가능성이 높음. R2/R3/R4를 먼저 전부 생성해
  디스크(Google Drive 등)에 저장한 뒤, 런타임을 재시작하고 학생 모델 학습으로 넘어갈 것.
- Colab 무료 T4는 세션 시간 제한이 있으므로, R2/R3/R4 생성 스크립트는 중간 체크포인트
  저장(예: N개 문서마다 저장) 로직을 반드시 포함할 것 — 10,340개 문서 × 3개 조건이라
  중단 시 처음부터 다시 돌리면 안 됨.

## 6. 지금 당장 해야 할 일 (우선순위 순)

1. **Colab에서 `notebooks/generate_teacher_data.ipynb` 실행** → `teacher_outputs.jsonl`
   생성. `data/unified.jsonl`을 먼저 Google Drive `MyDrive/ade-project/`에 업로드해야
   함(원문 포함이라 git에는 없음). `--limit 20`으로 먼저 속도 확인 후 전체 실행 —
   10,340건 × 3조건이라 세션 여러 번에 걸쳐 이어서 돌려야 할 가능성 높음(체크포인트
   재개 지원됨). 끝나면 `check_length_balance()` 리포트로 R2 대비 R3/R4가 ±20% 안에
   드는지 확인할 것 — 벗어나면 프롬프트 재검토 필요.
2. **Colab에서 `notebooks/train_qlora.ipynb` 실행** (teacher 생성과 다른 세션에서) →
   R0/R2/R3/R4는 forum·literature 양쪽 다 가능, R1은 `--cv-round 1`/`4`의 cadec_v1만.
   모델 크기(0.5b/1.5b/3b) × 조건 × 축 조합을 노트북 상단 설정 셀만 바꿔가며 반복.
3. **UMLS 승인 대기 중** — 승인 이메일 오면 사용자에게 확인 후 PsyTAR/PHEE SNOMED CT
   정규화 매칭 스크립트 작성 (엔티티 링킹, UMLS REST API 또는 QuickUMLS). 이게 끝나야
   R1이 주 축(forum↔literature) 실험에 들어갈 수 있음. 급하지 않으면 R1 없이 R0/R2/R3/R4로
   먼저 진행해도 됨 — R1은 H1/H2 가설에 필수가 아닌 참고 조건.
4. **`evaluate.py`를 실제 모델 출력에 연결하는 스크립트가 아직 없음** — 특히 R2(자유
   서술)/R3/R4의 모델 출력 텍스트에서 ADE 문자열 리스트를 뽑아내는 파싱 단계가 빠져
   있다. R0/R1은 줄바꿈 분리로 간단하지만, R2는 자유서술이라 파싱이 까다롭고 R3/R4는
   `event: <span>` 필드를 정규식으로 뽑으면 됨. 이 파싱 로직 + 추론 실행 스크립트를
   새로 만들어야 evaluate.py를 실제로 쓸 수 있다.
5. 보정(calibration, temperature scaling, analysis_plan.md §11) 스크립트 아직 없음.
6. 폴드 간 표준편차 vs 시드 간 표준편차 비교 (supplementary figure로 남길 것 — 이미
   계획에 포함된 저비용 실험, 폴드 4개 × 시드 3개 결과를 나란히 놓기만 하면 됨)

## 6-1. 알아둘 것 (오늘 세션에서 발견/수정한 것들)

- `.gitignore`의 `data/raw/`, `data/*.jsonl` 패턴은 원래 슬래시 때문에 저장소
  루트에만 적용되고 `research-project/data/` 하위는 못 걸렀던 버그가 있었음 →
  `**/data/*.jsonl`, `**/data/raw/` 등으로 수정함. 앞으로 `research-project/data/`
  아래 새 파생 파일을 만들 때는 원문 포함 여부를 반드시 확인하고, 필요하면
  `.gitignore` 패턴이 실제로 그 경로를 잡는지 `git check-ignore -v`로 확인할 것.
- `normalize_cadec_v1.py`는 오프셋이 아니라 원본 T-id(`T3`)와 sct 파일의
  `TT3`를 매칭한다 — CADEC v1 ADR의 15.9%가 불연속 스팬이라 오프셋 문자열
  비교는 프래그먼트 순서 차이로 깨질 수 있기 때문. 같은 패턴이 필요한 다른
  코퍼스 작업에도 참고할 것.
- 로컬 저장소에는 원본 zip을 `data/raw/`에 풀고 `build_unified.py`가 기대하는
  경로(`cadecv2_new/data/extracted/data`, `cadec/v2/cadec`, `psytar/psytar`,
  `PHEE-master/PHEE-master`)에 맞춰 심볼릭 링크로 연결해둔 `data/raw/build_root/`가
  있음. `ADE_DATA_ROOT=research-project/data/raw/build_root`로 실행하면 됨.

## 7. 판정 기준 (사전등록됨, 결과 보고 후 바꾸지 말 것)

- R2와 R3의 차이가 어느 방향으로든 측정되면 "성공"(=결과 있음)으로 간주. 차이 없음도 유효한 결론.
- 폴드별 승패 수로 일관성 보고 (예: "4개 폴드 중 3개에서 R3가 R2보다 높음")
- 계획 변경은 아래 3가지 경우에만 허용:
  ① 학습이 수렴하지 않아 수치 산출 불가
  ② 평가셋 200건 미만
  ③ 폴드 간 편차 때문에 어떤 조건 비교도 무의미

## 8. 하지 않기로 확정한 것

DPO, long context, multi-agent 시스템, memory 컴포넌트, 대형 지식그래프, test-time scaling.
스코프 크리프(그럴듯해 보이려고 기능을 늘리는 것)는 반복적으로 나타났던 유혹이므로 경계할 것.
"연구의 질은 시스템 복잡도가 아니라 깊이와 반증가능성에서 나온다"는 원칙을 계속 따를 것.

## 9. 작업 스타일 (참고)

- 반복적으로 문제를 파고들며, 프레이밍에 대해 적극적으로 되묻고 검증하는 것을 선호함
- 사실 주장(URL, 파일 형식, 데이터셋 통계 등)은 항상 1차 출처로 직접 검증하는 습관이 있음 —
  Claude가 틀렸을 때 스스로 발견해 정정한 이력이 여러 번 있음. 불확실하면 추측하지 말고
  확인이 필요하다고 명시할 것.
- 결정사항은 문서(analysis_plan.md 등)에 커밋해서 실행 중 스코프가 흔들리지 않게 하는 방식을 씀.
- 한국어로 소통 선호.

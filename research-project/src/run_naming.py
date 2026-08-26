"""어댑터 run_name 파싱 규칙 (run_inference.py / run_eval_all.py 공용, torch 의존성 없음)."""
import re

RUN_NAME_RE = re.compile(
    r"^(?P<size>0\.5b|1\.5b|3b)_(?P<cond>R[0-4])_(?P<domain>forum|literature)_seed(?P<seed>\d+)$"
)
OPPOSITE = {"forum": "literature", "literature": "forum"}

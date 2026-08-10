from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def make_safe_filename(original_filename: str) -> str:
    """업로드 파일명을 안전한 형태로 바꿉니다.

    이 함수는 과제 핵심이 아니므로 완성 코드로 제공합니다.
    """
    path = Path(original_filename)
    stem = re.sub(r"[^0-9a-zA-Z가-힣_.-]", "_", path.stem).strip("._") or "uploaded_file"
    suffix = path.suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{stem}{suffix}"


def get_file_type(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")

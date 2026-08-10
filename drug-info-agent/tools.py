"""openFDA drug label API를 사용하는 약물 정보 조회 도구."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


def _fetch_label(drug_name: str) -> dict | None:
    query = f'(openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}")'
    try:
        response = httpx.get(
            FDA_LABEL_URL,
            params={"search": query, "limit": 1},
            timeout=10,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    results = response.json().get("results", [])
    return results[0] if results else None


def _first(field: list[str] | None, fallback: str = "정보 없음") -> str:
    if not field:
        return fallback
    return field[0]


@tool
def search_drug_info(drug_name: str) -> str:
    """약물명(영문 성분명 또는 상품명)으로 효능/용법/주의사항을 openFDA에서 조회합니다."""
    label = _fetch_label(drug_name)
    if label is None:
        return f"'{drug_name}'에 대한 정보를 찾지 못했습니다."

    indications = _first(label.get("indications_and_usage"))
    dosage = _first(label.get("dosage_and_administration"))
    warnings = _first(label.get("warnings") or label.get("warnings_and_cautions"))

    return (
        f"[효능/효과]\n{indications[:500]}\n\n"
        f"[용법/용량]\n{dosage[:500]}\n\n"
        f"[주의사항]\n{warnings[:500]}"
    )


WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


@tool
def search_supplement_info(name: str) -> str:
    """건강기능식품/영양보충제(예: 이노시톨, 오메가3) 이름으로 일반 개요 정보를 Wikipedia에서 조회합니다.
    openFDA 의약품 라벨에는 없는 성분에 사용하세요. 의학적으로 검증된 임상 데이터가 아닌
    일반 참고 정보이므로, 상호작용/안전성 판단의 근거로 사용하면 안 됩니다."""
    try:
        response = httpx.get(
            WIKI_SUMMARY_URL.format(title=name.strip().replace(" ", "_")),
            headers={
                "User-Agent": (
                    "drug-info-agent/0.1 "
                    "(https://github.com/Gaeul5/Oracle_healthcare-bio_sLLM; "
                    "educational project)"
                )
            },
            timeout=10,
        )
        if response.status_code == 404:
            return f"'{name}'에 대한 일반 정보를 찾지 못했습니다."
        response.raise_for_status()
    except httpx.HTTPError:
        return f"'{name}' 조회 중 오류가 발생했습니다."

    extract = response.json().get("extract", "정보 없음")
    return f"[일반 참고 정보 - Wikipedia, 의학적 검증 아님]\n{extract[:600]}"


@tool
def check_drug_interaction(drug_a: str, drug_b: str) -> str:
    """두 약물(drug_a, drug_b, 영문 성분명) 간 상호작용 정보를 drug_a 라벨에서 조회합니다."""
    label = _fetch_label(drug_a)
    if label is None:
        return f"'{drug_a}'에 대한 라벨 정보를 찾지 못했습니다."

    interaction_text = _first(label.get("drug_interactions"), fallback="")
    if not interaction_text:
        return f"'{drug_a}' 라벨에 상호작용 정보가 없습니다."

    if drug_b.lower() in interaction_text.lower():
        return f"[{drug_a} 라벨 기준 {drug_b} 관련 상호작용]\n{interaction_text[:800]}"

    return (
        f"'{drug_a}' 라벨의 상호작용 항목에서 '{drug_b}'는 명시적으로 언급되지 않았습니다.\n"
        f"[{drug_a} 전체 상호작용 정보 일부]\n{interaction_text[:500]}"
    )


TOOLS = [search_drug_info, check_drug_interaction, search_supplement_info]

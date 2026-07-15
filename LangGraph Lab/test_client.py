import httpx
import asyncio

AGENT_A_URL = "http://localhost:8000"

TEST_CASES = [
    "3 + 5",
    "10 * 4",
    "100 / 5",
    "서울 날씨",
    "제주 날씨",
    "안녕하세요",   # 처리 불가 케이스
]

async def main():
    print("=" * 50)
    print("  A2A 통신 테스트 시작")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        for message in TEST_CASES:
            print(f"\n요청: '{message}'")
            try:
                resp = await client.post(
                    f"{AGENT_A_URL}/ask",
                    json={"message": message},
                    timeout=10.0,
                )
                data = resp.json()
                print(f"응답 (from {data['from_agent']}): {data['result']}")
            except Exception as e:
                print(f"에러: {e}")

    print("\n" + "=" * 50)
    print("  테스트 완료")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hr-assistant")

@mcp.tool()
def get_vacation_days(name: str) -> str:
    """직원의 남은 연차 일수를 조회합니다."""
    days = {"김철수": 7, "이영희": 12}
    return f"{name}님의 남은 연차: {days.get(name, 0)}일"

@mcp.resource("project://vacation")
def vacation_poliy() -> str:
    """휴가 규정 전문"""
    return open("vacation_policy.md",
                encoding= "utf-8").read()

@mcp.prompt()
def leave_request(name: str, dates: str) -> str:
    """휴가 신청서 초안 작성 프롬프트"""
    return (f"{name}의 {dates} 휴가 신청서를 "
            "격식 있는 사내 양식으로 작성해줘.")

if __name__ == "__main__":
    mcp.run()   #기본 transport: stdio
import sqlite3
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sales-db",
              host="0.0.0.0", port=8000)
DB = "saled.db"

@mcp.tool()
def query(sql: str) -> str:
    """읽기 전용 select 쿼리를 실행합니다."""
    if not sql.strip().lower().startswith("select"):
        return "오류: SELECT만 허용됩니다."
    conn = sqlite3.connect(DB)
    try:
        rows = conn.execute(sql).fetchall()
        return str(rows[:50])   #최대 50행
    finally:
        conn.close()

mcp.run(transport="streamable-http")
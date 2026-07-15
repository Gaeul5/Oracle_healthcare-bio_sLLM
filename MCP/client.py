import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(
    command="python", args=["server.py"])

async def main():
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            out = await sess.call_tool(
                "get_vacation_days", {"name": "김철수"})
            print(out.content[0].text)

asyncio.run(main())
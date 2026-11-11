import asyncio
import pytest
from langchain_core.messages import ToolMessage
from agents.researcher.utils import execute_tool_safely


class _MockTool:
    name = "mock_search"

    async def ainvoke(self, args):
        if args.get("sleep"):
            await asyncio.sleep(args["sleep"])
        if args.get("fail"):
            raise ValueError("simulated upstream error")
        return f"result:{args.get('query', '')}"


@pytest.mark.anyio
async def test_execute_tool_safely_handles_none_timeout_and_errors():
    missing = await execute_tool_safely(None, {"query": "test"})
    assert "unknown or unregistered tool" in missing

    ok = await execute_tool_safely(_MockTool(), {"query": "langgraph"}, timeout=2.0)
    assert ok == "result:langgraph"

    err = await execute_tool_safely(_MockTool(), {"fail": True}, timeout=2.0)
    assert "Error executing tool: simulated upstream error" in err

    timed_out = await execute_tool_safely(_MockTool(), {"sleep": 0.5}, timeout=0.02)
    assert "timed out" in timed_out

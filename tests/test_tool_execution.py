import asyncio
import pytest
from langchain_core.messages import ToolMessage
from agents.researcher.utils import (
    anthropic_websearch_called,
    execute_tool_safely,
    openai_websearch_called,
)


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

    bad_timeout = await execute_tool_safely(_MockTool(), {"query": "x"}, timeout=-1)
    assert "invalid timeout" in bad_timeout
    bool_timeout = await execute_tool_safely(_MockTool(), {"query": "x"}, timeout=True)
    assert "invalid timeout" in bool_timeout


def test_websearch_called_guards_malformed_items_and_booleans():
    class _Resp:
        def __init__(self, response_metadata=None, additional_kwargs=None):
            self.response_metadata = response_metadata or {}
            self.additional_kwargs = additional_kwargs or {}

    assert anthropic_websearch_called(_Resp({"usage": {"server_tool_use": {"web_search_requests": True}}})) is False
    assert anthropic_websearch_called(_Resp({"usage": {"server_tool_use": {"web_search_requests": 2}}})) is True
    assert openai_websearch_called(_Resp(additional_kwargs={"tool_outputs": ["noise", {"type": "web_search_call"}]})) is True

import asyncio
import pytest
from langchain_core.messages import ToolMessage
from agents.researcher.utils import (
    anthropic_websearch_called,
    execute_tool_safely,
    gemini_websearch_called,
    is_websearch_called,
    normalize_and_deduplicate_sources,
    openai_websearch_called,
    sanitize_search_query,
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


@pytest.mark.anyio
async def test_execute_tool_safely_sync_and_callable():
    def sync_tool(args):
        return f"sync:{args.get('v')}"

    class SyncInvokable:
        name = "sync_invokable"
        def invoke(self, args):
            return f"invoke:{args.get('v')}"

    res_fn = await execute_tool_safely(sync_tool, {"v": 123}, timeout=1.0)
    assert res_fn == "sync:123"

    res_inv = await execute_tool_safely(SyncInvokable(), {"v": 456}, timeout=1.0)
    assert res_inv == "invoke:456"

    not_callable = await execute_tool_safely("not_a_tool", {"v": 1})
    assert "neither invokable nor callable" in not_callable


def test_openai_websearch_called_tool_calls_and_none():
    assert openai_websearch_called(None) is False

    class _ToolCallResp:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    assert openai_websearch_called(_ToolCallResp([{"name": "web_search_preview"}])) is True
    assert openai_websearch_called(_ToolCallResp([{"name": "other_tool"}])) is False


def test_gemini_and_unified_websearch_called():
    assert gemini_websearch_called(None) is False
    assert is_websearch_called(None) is False

    class _GeminiMetaResp:
        def __init__(self, response_metadata=None, additional_kwargs=None, tool_calls=None):
            self.response_metadata = response_metadata or {}
            self.additional_kwargs = additional_kwargs or {}
            self.tool_calls = tool_calls or []

    # Grounding metadata in response_metadata
    r1 = _GeminiMetaResp(response_metadata={"grounding_metadata": {"web_search_sources": ["url1"]}})
    assert gemini_websearch_called(r1) is True
    assert is_websearch_called(r1) is True

    # Grounding metadata in additional_kwargs
    r2 = _GeminiMetaResp(additional_kwargs={"groundingMetadata": {"queries": ["q"]}})
    assert gemini_websearch_called(r2) is True

    # Tool calls with google_search
    r3 = _GeminiMetaResp(tool_calls=[{"name": "google_search", "args": {"q": "langgraph"}}])
    assert gemini_websearch_called(r3) is True
    assert is_websearch_called(r3) is True

    # Negative case
    r4 = _GeminiMetaResp()
    assert gemini_websearch_called(r4) is False
    assert is_websearch_called(r4) is False


def test_sanitize_search_query():
    # Null and empty
    assert sanitize_search_query(None) == ""
    assert sanitize_search_query("") == ""
    assert sanitize_search_query("   ") == ""
    assert sanitize_search_query(12345) == ""
    assert sanitize_search_query("query", max_length=0) == ""

    # Whitespace and control character stripping
    raw = "  \x00langchain\x1f \t\n  graph   deep\r\nresearch  \x7f "
    assert sanitize_search_query(raw) == "langchain graph deep research"

    # Truncation at word boundary
    long_query = "autonomous multi-agent system state machine orchestrator architecture design pattern"
    res = sanitize_search_query(long_query, max_length=45)
    assert len(res) <= 45
    assert not res.endswith(" ")
    assert res == "autonomous multi-agent system state machine"


def test_normalize_and_deduplicate_sources():
    # Empty inputs
    assert normalize_and_deduplicate_sources(None) == []
    assert normalize_and_deduplicate_sources([]) == []
    assert normalize_and_deduplicate_sources(["", "   ", None]) == []

    # Scheme / host lowercase and standard port removal
    raw_sources = [
        "HTTP://EXAMPLE.COM:80/path/to/doc?utm_source=twitter&topic=graph#section1",
        "https://example.com/path/to/doc?topic=graph",
        {"url": "https://example.com/path/to/doc/?topic=graph&fbclid=123", "title": "Doc Title", "score": 0.95},
        {"url": "https://other.org/article", "title": "Other Article"},
    ]
    deduped = normalize_and_deduplicate_sources(raw_sources)
    assert len(deduped) == 3

    # Check canonical URL for example.com
    ex_items = [d for d in deduped if "example.com" in d["url"]]
    assert len(ex_items) == 2
    # The one with https should have topic=graph and metadata merged
    https_item = next(d for d in ex_items if d["url"].startswith("https://"))
    assert https_item["url"] == "https://example.com/path/to/doc?topic=graph"
    assert https_item["title"] == "Doc Title"
    assert https_item["score"] == 0.95

    # Check other.org
    other_item = next(d for d in deduped if "other.org" in d["url"])
    assert other_item["url"] == "https://other.org/article"
    assert other_item["title"] == "Other Article"



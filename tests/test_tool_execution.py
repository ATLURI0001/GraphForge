import asyncio
import pytest
from langchain_core.messages import ToolMessage
from agents.researcher.utils import (
    anthropic_websearch_called,
    execute_tool_safely,
    calculate_citation_diversity,
    extract_academic_identifiers,
    filter_sources_by_allowed_domains,
    format_citation_markdown,
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


def test_format_citation_markdown():
    sources = [
        {"url": "https://example.com/a", "title": "Alpha [Guide]"},
        {"url": "https://example.com/b", "title": "Beta Overview"},
        {"url": "https://example.com/c"},  # missing title fallback
    ]
    formatted = format_citation_markdown(sources, max_citations=2)
    lines = formatted.splitlines()
    assert len(lines) == 2
    assert lines[0] == r"[1] [Alpha \[Guide\]](https://example.com/a)"
    assert lines[1] == "[2] [Beta Overview](https://example.com/b)"

    # Empty fallback
    assert format_citation_markdown([]) == ""
    assert format_citation_markdown(None) == ""


def test_calculate_citation_diversity():
    sources = [
        {"url": "https://arxiv.org/abs/1234"},
        {"url": "https://arxiv.org/pdf/1234.pdf"},
        {"url": "https://github.com/repo/test"},
        {"url": "https://nature.com/articles/456"},
    ]
    metrics = calculate_citation_diversity(sources)
    assert metrics["total_sources"] == 4
    assert metrics["unique_domains"] == 3
    assert metrics["domain_distribution"] == {
        "arxiv.org": 2,
        "github.com": 1,
        "nature.com": 1,
    }
    assert metrics["diversity_ratio"] == 0.75

    empty_metrics = calculate_citation_diversity([])
    assert empty_metrics["total_sources"] == 0
    assert empty_metrics["diversity_ratio"] == 0.0


def test_filter_sources_by_allowed_domains():
    sources = [
        {"url": "https://arxiv.org/abs/1234", "title": "Paper 1"},
        {"url": "https://en.wikipedia.org/wiki/Test", "title": "Wiki"},
        {"url": "https://untrusted.com/blog", "title": "Untrusted"},
        {"url": "not-a-url"},
        "invalid-item",
    ]
    filtered = filter_sources_by_allowed_domains(
        sources, ["arxiv.org", "wikipedia.org"], allow_subdomains=True
    )
    assert len(filtered) == 2
    assert [s["title"] for s in filtered] == ["Paper 1", "Wiki"]

    strict = filter_sources_by_allowed_domains(
        sources, ["wikipedia.org"], allow_subdomains=False
    )
    assert len(strict) == 0

    assert filter_sources_by_allowed_domains([], ["arxiv.org"]) == []
    assert filter_sources_by_allowed_domains(None, ["arxiv.org"]) == []
    assert filter_sources_by_allowed_domains(sources, []) == []
    assert filter_sources_by_allowed_domains(sources, None) == []


def test_extract_academic_identifiers():
    text = (
        "References: 10.1038/s41586-020-2649-2 and arXiv:2305.18290v2. "
        "Also see pmid: 32840742 and duplicate 10.1038/s41586-020-2649-2."
    )
    res = extract_academic_identifiers(text)
    assert res["dois"] == ["10.1038/s41586-020-2649-2"]
    assert res["arxiv_ids"] == ["2305.18290v2"]
    assert res["pmids"] == ["32840742"]

    assert extract_academic_identifiers(None) == {
        "dois": [],
        "arxiv_ids": [],
        "pmids": [],
    }
    assert extract_academic_identifiers("") == {
        "dois": [],
        "arxiv_ids": [],
        "pmids": [],
    }




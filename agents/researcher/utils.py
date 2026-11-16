"""Simplified utility functions for the Deep Research agent."""

import json
import logging
import math
import re
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from langchain_core.messages import (
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.tools import tool

from agents.researcher.models import ResearchComplete

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress.

    Use this tool after each search to analyze results and plan next steps.

    Args:
        reflection: Detailed reflection on research progress and next steps

    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"


##########################
# Tool Utils
##########################

async def get_all_tools():
    """Assemble complete toolkit for research operations.

    Returns tools including OpenAI's native web search.
    """
    # Core research tools
    tools = [tool(ResearchComplete), think_tool]

    # Add OpenAI's native web search
    # This is a special tool definition that OpenAI models recognize
    tools.append({"type": "web_search_preview"})

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]) -> list[str]:
    """Extract notes from tool call messages with defensive handling."""
    if not messages:
        return []

    extracted: list[str] = []
    seen: set[str] = set()

    for tool_msg in filter_messages(messages, include_types="tool"):
        raw_content = getattr(tool_msg, "content", None)
        if raw_content is None:
            continue
        if isinstance(raw_content, str):
            text = raw_content.strip()
        elif isinstance(raw_content, (dict, list)):
            text = json.dumps(raw_content, sort_keys=True, default=str)
        else:
            text = str(raw_content).strip()

        if text and text not in seen:
            seen.add(text)
            extracted.append(text)

    return extracted


##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used."""
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        web_search_requests = server_tool_use.get("web_search_requests")
        if (
            web_search_requests is None
            or isinstance(web_search_requests, bool)
            or not isinstance(web_search_requests, (int, float))
        ):
            return False

        return web_search_requests > 0

    except (AttributeError, TypeError):
        return False


def openai_websearch_called(response):
    """Detect if OpenAI's web search was used."""
    if response is None:
        return False
    try:
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        tool_outputs = additional_kwargs.get("tool_outputs")
        if isinstance(tool_outputs, list):
            for tool_output in tool_outputs:
                if isinstance(tool_output, dict) and tool_output.get("type") == "web_search_call":
                    return True

        tool_calls = getattr(response, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and call.get("name") in ("web_search_preview", "web_search"):
                    return True

        return False
    except (AttributeError, TypeError):
        return False


def gemini_websearch_called(response) -> bool:
    """Detect if Google Gemini's native web search grounding was used."""
    if response is None:
        return False
    try:
        meta = getattr(response, "response_metadata", {}) or {}
        if meta.get("grounding_metadata") or meta.get("groundingMetadata"):
            return True
        kwargs = getattr(response, "additional_kwargs", {}) or {}
        if kwargs.get("grounding_metadata") or kwargs.get("groundingMetadata"):
            return True
        tool_calls = getattr(response, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and call.get("name") in ("google_search", "google_search_retrieval"):
                    return True
        return False
    except (AttributeError, TypeError):
        return False


def is_websearch_called(response) -> bool:
    """Unified detector for provider-native web search usage (Anthropic, OpenAI, Gemini)."""
    return (
        anthropic_websearch_called(response)
        or openai_websearch_called(response)
        or gemini_websearch_called(response)
    )


async def execute_tool_safely(tool, args, timeout: float | None = 30.0):
    """Safely execute a tool with missing-tool, timeout, sync/async invocation, and error handling."""
    if tool is None:
        return "Error executing tool: unknown or unregistered tool requested"
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        return f"Error executing tool: invalid timeout {timeout!r}"
    import asyncio
    try:
        if hasattr(tool, "ainvoke"):
            coro = tool.ainvoke(args)
        elif hasattr(tool, "invoke"):
            coro = asyncio.to_thread(tool.invoke, args)
        elif callable(tool):
            coro = asyncio.to_thread(tool, args)
        else:
            return "Error executing tool: tool is neither invokable nor callable"

        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro
    except asyncio.TimeoutError:
        tool_name = getattr(tool, "name", repr(tool))
        return f"Error executing tool '{tool_name}': timed out after {timeout}s"
    except Exception as e:
        return f"Error executing tool: {str(e)}"


##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def sanitize_search_query(query: str | None, max_length: int = 400) -> str:
    """Sanitize and normalize a raw web search query string.

    Performs control character stripping, whitespace normalization, and safe boundary truncation.

    Args:
        query: Raw search query string.
        max_length: Maximum allowed character length for the query.

    Returns:
        Sanitized, normalized query string.
    """
    if not query or not isinstance(query, str):
        return ""
    if max_length <= 0:
        return ""

    # Strip non-printable / control characters (ASCII 0x00-0x1F and 0x7F-0x9F) except space
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", query)
    # Collapse multiple whitespace sequences into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) <= max_length:
        return cleaned

    # Truncate at word boundary if possible within the latter half of max_length
    truncated = cleaned[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        return truncated[:last_space].strip()
    return truncated.strip()


def normalize_and_deduplicate_sources(
    sources: list[dict | str] | None,
    default_title: str = "Untitled Source",
) -> list[dict]:
    """Normalize and deduplicate web sources and citations.

    Strips URL fragments, normalizes scheme/host casing, removes common tracking query parameters
    (e.g., utm_*, ref, fbclid), and merges metadata across identical canonical URLs.

    Args:
        sources: List of string URLs or source dictionaries containing at least a 'url' key.
        default_title: Default title applied when a source lacks a title.

    Returns:
        List of canonicalized, unique source dictionaries.
    """
    if not sources:
        return []

    tracking_params = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "fbclid",
        "gclid",
        "ocid",
        "mc_cid",
        "mc_eid",
    }

    seen: dict[str, dict] = {}

    for item in sources:
        if not item:
            continue
        if isinstance(item, str):
            raw_url = item.strip()
            data = {"url": raw_url, "title": default_title}
        elif isinstance(item, dict):
            raw_url = str(item.get("url", "")).strip()
            data = dict(item)
            if not data.get("title"):
                data["title"] = default_title
        else:
            continue

        if not raw_url:
            continue

        try:
            parsed = urlparse(raw_url)
            if not parsed.scheme or not parsed.netloc:
                continue

            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()

            # Remove standard default port
            if scheme == "http" and netloc.endswith(":80"):
                netloc = netloc[:-3]
            elif scheme == "https" and netloc.endswith(":443"):
                netloc = netloc[:-4]

            path = parsed.path or "/"
            if len(path) > 1 and path.endswith("/"):
                path = path.rstrip("/")

            # Filter tracking query parameters
            query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
            filtered_pairs = [(k, v) for k, v in query_pairs if k.lower() not in tracking_params]
            clean_query = urlencode(filtered_pairs)

            canonical_url = urlunparse((scheme, netloc, path, "", clean_query, ""))

            if canonical_url not in seen:
                data["url"] = canonical_url
                seen[canonical_url] = data
            else:
                # Merge metadata: prefer non-default title, retain additional keys
                existing = seen[canonical_url]
                if existing.get("title") == default_title and data.get("title") != default_title:
                    existing["title"] = data["title"]
                for k, v in data.items():
                    if k not in existing and v is not None:
                        existing[k] = v
        except Exception:
            continue

    return list(seen.values())


def format_citation_markdown(
    sources: list[dict] | None,
    max_citations: int = 20,
) -> str:
    """Format canonical sources into numbered Markdown bibliographic citations.

    Args:
        sources: List of source dictionaries containing 'url' and optionally 'title'.
        max_citations: Maximum citations to render (clamped between 1 and 100).

    Returns:
        Formatted markdown citation text.
    """
    if not sources or not isinstance(sources, list):
        return ""
    if max_citations <= 0:
        return ""
    max_citations = min(100, max_citations)

    lines: list[str] = []
    count = 0
    for item in sources:
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url", "")).strip()
        if not raw_url:
            continue
        title = str(item.get("title") or "Untitled Source").strip()
        # Escape markdown square brackets in title
        escaped_title = title.replace("[", r"\[").replace("]", r"\]")
        count += 1
        lines.append(f"[{count}] [{escaped_title}]({raw_url})")
        if count >= max_citations:
            break

    return "\n".join(lines)


def calculate_citation_diversity(sources: list[dict] | None) -> dict:
    """Calculate domain distribution and diversity metrics across research sources.

    Args:
        sources: List of source dictionaries with 'url' keys.

    Returns:
        Dictionary with total_sources, unique_domains, domain_distribution, and diversity_ratio.
    """
    if not sources or not isinstance(sources, list):
        return {
            "total_sources": 0,
            "unique_domains": 0,
            "domain_distribution": {},
            "diversity_ratio": 0.0,
        }

    distribution: dict[str, int] = {}
    total = 0

    for item in sources:
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url", "")).strip()
        if not raw_url:
            continue
        try:
            parsed = urlparse(raw_url)
            host = parsed.netloc.lower()
            if not host:
                continue
            if ":" in host:
                host = host.split(":", 1)[0]
            distribution[host] = distribution.get(host, 0) + 1
            total += 1
        except Exception:
            continue

    unique_domains = len(distribution)
    ratio = round(unique_domains / total, 2) if total > 0 else 0.0

    return {
        "total_sources": total,
        "unique_domains": unique_domains,
        "domain_distribution": distribution,
        "diversity_ratio": ratio,
    }

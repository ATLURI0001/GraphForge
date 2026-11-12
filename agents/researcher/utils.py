"""Simplified utility functions for the Deep Research agent."""

import json
import logging
import math
from datetime import datetime

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
    try:
        tool_outputs = response.additional_kwargs.get("tool_outputs")
        if not tool_outputs:
            return False

        for tool_output in tool_outputs:
            if isinstance(tool_output, dict) and tool_output.get("type") == "web_search_call":
                return True

        return False
    except (AttributeError, TypeError):
        return False


async def execute_tool_safely(tool, args, timeout: float | None = 30.0):
    """Safely execute a tool with missing-tool, timeout, and error handling."""
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
        coro = tool.ainvoke(args)
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

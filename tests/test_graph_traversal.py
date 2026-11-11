import pytest
from langchain_core.messages import ToolMessage
from utils.utils import (
    GraphCycleError,
    GraphRecursionLimitError,
    find_cycles,
    validate_graph_traversal,
)
from agents.researcher.utils import get_notes_from_tool_calls


def test_find_cycles_acyclic():
    graph = {
        "start": ["node_a", "node_b"],
        "node_a": ["node_c"],
        "node_b": ["node_c"],
        "node_c": ["end"],
        "end": [],
    }
    cycles = find_cycles(graph)
    assert cycles == []


def test_find_cycles_detected():
    graph = {
        "node_a": ["node_b"],
        "node_b": ["node_c"],
        "node_c": ["node_a"],
    }
    cycles = find_cycles(graph)
    assert len(cycles) > 0
    cycle = cycles[0]
    assert cycle[0] == cycle[-1]


def test_validate_graph_traversal_acyclic():
    graph = {
        "start": ["step1"],
        "step1": ["step2"],
        "step2": ["end"],
        "end": [],
    }
    result = validate_graph_traversal("start", graph, max_depth=10)
    assert result["max_path_depth"] == 4
    assert "start" in result["reachable_nodes"]
    assert "end" in result["reachable_nodes"]
    assert result["cycles_found"] == []


def test_validate_graph_traversal_cycle_disallowed():
    graph = {
        "start": ["loop1"],
        "loop1": ["loop2"],
        "loop2": ["loop1"],
    }
    with pytest.raises(GraphCycleError) as exc_info:
        validate_graph_traversal("start", graph, max_depth=10, allow_cycles=False)
    assert "Prohibited cycle detected" in str(exc_info.value)


def test_validate_graph_traversal_recursion_limit():
    graph = {
        "start": ["loop1"],
        "loop1": ["loop2"],
        "loop2": ["loop1"],
    }
    # Allow cycle, but set depth to exceed limit
    with pytest.raises(GraphRecursionLimitError) as exc_info:
        validate_graph_traversal("start", graph, max_depth=3, allow_cycles=True)
    assert "exceeded max depth of 3" in str(exc_info.value)


def test_validate_graph_traversal_invalid_depth():
    with pytest.raises(ValueError):
        validate_graph_traversal("start", {}, max_depth=0)


def test_get_notes_from_tool_calls_defensive():
    assert get_notes_from_tool_calls([]) == []
    assert get_notes_from_tool_calls(None) == []

    msg1 = ToolMessage(content="Note 1", tool_call_id="call_1")
    msg2 = ToolMessage(content="Note 2", tool_call_id="call_2")
    msg_dup = ToolMessage(content="Note 1", tool_call_id="call_3")
    msg_json = ToolMessage(content={"summary": "ok"}, tool_call_id="call_4")

    notes = get_notes_from_tool_calls([msg1, msg2, msg_dup, msg_json])
    assert notes[:2] == ["Note 1", "Note 2"]
    assert "summary" in notes[2] and "ok" in notes[2]
    assert len(notes) == 3

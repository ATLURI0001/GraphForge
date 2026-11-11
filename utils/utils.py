import sqlite3
from typing import Optional
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with ASCII fallback.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
        xray: Whether to show the internal structure of the graph
    """
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"⚠️  Image rendering failed: {e}")
        print("\n📊 Showing ASCII diagram instead:\n")
        ascii_diagram = graph.get_graph(xray=xray).draw_ascii()
        print(ascii_diagram)
        return None

def get_engine_for_chinook_db(timeout: int = 15, sql_script: Optional[str] = None):
    """Pull sql file, populate in-memory database, and create engine.
    
    Args:
        timeout: Network timeout in seconds for fetching the schema
        sql_script: Optional pre-loaded SQL script to bypass remote network calls
    """
    if sql_script is None:
        url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


class GraphRecursionLimitError(RuntimeError):
    """Raised when graph execution exceeds the maximum allowable recursion depth."""
    pass


class GraphCycleError(ValueError):
    """Raised when an unhandled or prohibited cycle is detected in a graph workflow."""
    pass


def find_cycles(adjacency_list: dict[str, list[str]]) -> list[list[str]]:
    """Detect all simple cycles in a directed graph.

    Args:
        adjacency_list: Mapping from node identifier to list of successor node identifiers.

    Returns:
        List of detected cycles, where each cycle is represented as a list of node identifiers.
    """
    visited: set[str] = set()
    rec_stack: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str):
        visited.add(node)
        rec_stack.append(node)

        for neighbor in adjacency_list.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                cycle_start = rec_stack.index(neighbor)
                cycle = rec_stack[cycle_start:] + [neighbor]
                cycles.append(cycle)

        rec_stack.pop()

    for n in list(adjacency_list.keys()):
        if n not in visited:
            dfs(n)

    return cycles


def validate_graph_traversal(
    start_node: str,
    adjacency_list: dict[str, list[str]],
    max_depth: int = 50,
    allow_cycles: bool = False,
) -> dict:
    """Validate that graph traversal starting from start_node terminates safely.

    Args:
        start_node: The initial node to begin traversal from.
        adjacency_list: Directed adjacency mapping.
        max_depth: Maximum path length allowed before triggering recursion limit error.
        allow_cycles: Whether controlled cycles are permitted if depth stays within max_depth.

    Returns:
        Dict with 'reachable_nodes', 'max_path_depth', and 'cycles_found'.

    Raises:
        GraphCycleError: If allow_cycles is False and a cycle is detected.
        GraphRecursionLimitError: If traversal exceeds max_depth.
    """
    if max_depth <= 0:
        raise ValueError(f"max_depth must be positive, got {max_depth}")

    cycles = find_cycles(adjacency_list)
    if not allow_cycles and cycles:
        cycle_str = " -> ".join(cycles[0])
        raise GraphCycleError(f"Prohibited cycle detected in graph: {cycle_str}")

    visited_nodes: set[str] = set()
    max_observed_depth = 0

    # BFS/DFS traversal checking path depth
    stack: list[tuple[str, int, list[str]]] = [(start_node, 1, [start_node])]

    while stack:
        current, depth, path = stack.pop()
        visited_nodes.add(current)
        if depth > max_observed_depth:
            max_observed_depth = depth

        if depth > max_depth:
            path_str = " -> ".join(path)
            raise GraphRecursionLimitError(
                f"Graph traversal exceeded max depth of {max_depth}. Path: {path_str}"
            )

        for neighbor in adjacency_list.get(current, []):
            if allow_cycles:
                # In allowed cycles, avoid infinite stack explosion by bounding per-branch repetitions
                if path.count(neighbor) > 2:
                    continue
            stack.append((neighbor, depth + 1, path + [neighbor]))

    return {
        "reachable_nodes": sorted(list(visited_nodes)),
        "max_path_depth": max_observed_depth,
        "cycles_found": cycles,
    }


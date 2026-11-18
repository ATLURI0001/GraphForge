"""DAG topological wave scheduler, cycle diagnostics, and async executor for research subtasks."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class CyclicDependencyError(ValueError):
    """Raised when a research task graph contains one or more dependency cycles."""

    def __init__(self, cycle_path: list[str]):
        self.cycle_path = cycle_path
        chain = " -> ".join(cycle_path)
        super().__init__(f"Cyclic task dependency detected: {chain}")


class UnknownDependencyError(ValueError):
    """Raised when a task references an undeclared upstream dependency."""


@dataclass(frozen=True)
class ResearchTask:
    """Immutable descriptor of a research subtask node in an execution DAG."""

    task_id: str
    depends_on: tuple[str, ...] = ()
    estimated_cost_ms: float = 100.0
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if self.estimated_cost_ms < 0:
            raise ValueError("estimated_cost_ms must be non-negative")


class DAGExecutionPlanner:
    """Topological wave planner, critical-path analyzer, and bounded async executor."""

    def __init__(self, tasks: list[ResearchTask]) -> None:
        self._tasks: dict[str, ResearchTask] = {}
        for task in tasks:
            if task.task_id in self._tasks:
                raise ValueError(f"Duplicate task_id in DAG: '{task.task_id}'")
            self._tasks[task.task_id] = task

        self._adj: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}

        for tid, task in self._tasks.items():
            seen_deps: set[str] = set()
            for dep in task.depends_on:
                if dep not in self._tasks:
                    raise UnknownDependencyError(
                        f"Task '{tid}' depends on unknown task '{dep}'"
                    )
                if dep == tid:
                    raise CyclicDependencyError([tid, tid])
                if dep not in seen_deps:
                    seen_deps.add(dep)
                    self._adj[dep].append(tid)
                    self._in_degree[tid] += 1

        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """Detect cycles via 3-color DFS and extract the exact back-edge cycle chain."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        def dfs(node: str) -> None:
            color[node] = GRAY
            for nxt in sorted(self._adj.get(node, [])):
                if color[nxt] == GRAY:
                    cycle = [nxt]
                    cur: str | None = node
                    while cur is not None and cur != nxt:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(nxt)
                    cycle.reverse()
                    raise CyclicDependencyError(cycle)
                if color[nxt] == WHITE:
                    parent[nxt] = node
                    dfs(nxt)
            color[node] = BLACK

        for tid in sorted(self._tasks):
            if color[tid] == WHITE:
                dfs(tid)

    def topological_order(self) -> list[str]:
        """Return a deterministic priority-weighted topological ordering of all tasks."""
        waves = self.plan_waves()
        return [tid for wave in waves for tid in wave]

    def plan_waves(self, max_parallelism: int | None = None) -> list[list[str]]:
        """Partition tasks into parallelizable execution waves using Kahn's algorithm.

        Tasks within each wave have zero mutual dependencies and are ordered by
        descending priority, then lexicographically by task_id.
        """
        if max_parallelism is not None and max_parallelism < 1:
            raise ValueError("max_parallelism must be >= 1")

        in_deg = dict(self._in_degree)
        ready = sorted(
            [tid for tid, deg in in_deg.items() if deg == 0],
            key=lambda tid: (-self._tasks[tid].priority, tid),
        )

        waves: list[list[str]] = []
        processed = 0

        while ready:
            current_wave = ready
            next_ready: list[str] = []

            if max_parallelism is not None and len(current_wave) > max_parallelism:
                waves.extend(
                    current_wave[i : i + max_parallelism]
                    for i in range(0, len(current_wave), max_parallelism)
                )
            else:
                waves.append(current_wave)

            for tid in current_wave:
                processed += 1
                for child in self._adj.get(tid, []):
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        next_ready.append(child)

            ready = sorted(
                next_ready,
                key=lambda tid: (-self._tasks[tid].priority, tid),
            )

        if processed != len(self._tasks):
            self._assert_acyclic()

        return waves

    def critical_path(self) -> tuple[list[str], float]:
        """Compute the longest-cost dependency chain (Critical Path Method)."""
        order = self.topological_order()
        if not order:
            return [], 0.0

        dist: dict[str, float] = {}
        predecessor: dict[str, str | None] = {}

        for tid in order:
            task = self._tasks[tid]
            best_prev_cost = 0.0
            best_prev_node: str | None = None
            for dep in task.depends_on:
                if dist[dep] > best_prev_cost:
                    best_prev_cost = dist[dep]
                    best_prev_node = dep
            dist[tid] = best_prev_cost + task.estimated_cost_ms
            predecessor[tid] = best_prev_node

        end_node = max(order, key=lambda tid: (dist[tid], tid))
        path: list[str] = []
        cur: str | None = end_node
        while cur is not None:
            path.append(cur)
            cur = predecessor[cur]
        path.reverse()
        return path, round(dist[end_node], 3)

    async def execute_dag(
        self,
        runner: Callable[[str], Awaitable[Any]],
        max_concurrency: int = 4,
    ) -> dict[str, dict[str, Any]]:
        """Execute the DAG asynchronously wave-by-wave with bounded concurrency.

        If an upstream task raises an exception, downstream dependents are
        automatically skipped with status 'skipped_upstream_failed'.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        waves = self.plan_waves()
        sem = asyncio.Semaphore(max_concurrency)
        outcomes: dict[str, dict[str, Any]] = {}

        async def _run_node(tid: str) -> None:
            failed_deps = [
                dep
                for dep in self._tasks[tid].depends_on
                if outcomes.get(dep, {}).get("status") != "completed"
            ]
            if failed_deps:
                outcomes[tid] = {
                    "status": "skipped_upstream_failed",
                    "result": None,
                    "error": f"Blocked by failed dependencies: {', '.join(sorted(failed_deps))}",
                }
                return

            async with sem:
                try:
                    res = await runner(tid)
                    outcomes[tid] = {"status": "completed", "result": res, "error": None}
                except Exception as exc:
                    outcomes[tid] = {"status": "failed", "result": None, "error": str(exc)}

        for wave in waves:
            await asyncio.gather(*(_run_node(tid) for tid in wave))

        return outcomes

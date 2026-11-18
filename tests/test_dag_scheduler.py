import asyncio
import pytest

from agents.researcher.scheduler import (
    CyclicDependencyError,
    DAGExecutionPlanner,
    ResearchTask,
    UnknownDependencyError,
)


def test_dag_wave_planning_and_priority_ordering():
    tasks = [
        ResearchTask("plan", estimated_cost_ms=50, priority=10),
        ResearchTask("fetch_arxiv", depends_on=("plan",), estimated_cost_ms=300, priority=5),
        ResearchTask("fetch_docs", depends_on=("plan",), estimated_cost_ms=120, priority=9),
        ResearchTask("fetch_patents", depends_on=("plan",), estimated_cost_ms=200, priority=1),
        ResearchTask("synthesize", depends_on=("fetch_arxiv", "fetch_docs", "fetch_patents"), estimated_cost_ms=150),
    ]
    planner = DAGExecutionPlanner(tasks)

    waves = planner.plan_waves()
    assert waves == [
        ["plan"],
        ["fetch_docs", "fetch_arxiv", "fetch_patents"],
        ["synthesize"],
    ]

    chunked_waves = planner.plan_waves(max_parallelism=2)
    assert chunked_waves == [
        ["plan"],
        ["fetch_docs", "fetch_arxiv"],
        ["fetch_patents"],
        ["synthesize"],
    ]

    path, cost = planner.critical_path()
    assert path == ["plan", "fetch_arxiv", "synthesize"]
    assert cost == 500.0


def test_dag_detects_cycles_with_exact_chain_and_unknown_deps():
    with pytest.raises(CyclicDependencyError) as exc_info:
        DAGExecutionPlanner(
            [
                ResearchTask("a", depends_on=("c",)),
                ResearchTask("b", depends_on=("a",)),
                ResearchTask("c", depends_on=("b",)),
            ]
        )
    assert exc_info.value.cycle_path[0] == exc_info.value.cycle_path[-1]
    assert set(exc_info.value.cycle_path) == {"a", "b", "c"}

    with pytest.raises(CyclicDependencyError):
        DAGExecutionPlanner([ResearchTask("self_loop", depends_on=("self_loop",))])

    with pytest.raises(UnknownDependencyError):
        DAGExecutionPlanner([ResearchTask("orphan", depends_on=("missing",))])


@pytest.mark.anyio
async def test_dag_async_execution_and_upstream_failure_isolation():
    tasks = [
        ResearchTask("root"),
        ResearchTask("branch_ok", depends_on=("root",)),
        ResearchTask("branch_fail", depends_on=("root",)),
        ResearchTask("downstream_ok", depends_on=("branch_ok",)),
        ResearchTask("downstream_blocked", depends_on=("branch_ok", "branch_fail")),
    ]
    planner = DAGExecutionPlanner(tasks)

    active = 0
    peak_concurrency = 0

    async def runner(tid: str):
        nonlocal active, peak_concurrency
        active += 1
        peak_concurrency = max(peak_concurrency, active)
        try:
            await asyncio.sleep(0.01)
            if tid == "branch_fail":
                raise RuntimeError("upstream timeout")
            return f"ok:{tid}"
        finally:
            active -= 1

    outcomes = await planner.execute_dag(runner, max_concurrency=2)
    assert peak_concurrency <= 2
    assert outcomes["root"]["status"] == "completed"
    assert outcomes["branch_ok"]["status"] == "completed"
    assert outcomes["branch_fail"]["status"] == "failed"
    assert "upstream timeout" in outcomes["branch_fail"]["error"]
    assert outcomes["downstream_ok"]["status"] == "completed"
    assert outcomes["downstream_blocked"]["status"] == "skipped_upstream_failed"
    assert "branch_fail" in outcomes["downstream_blocked"]["error"]

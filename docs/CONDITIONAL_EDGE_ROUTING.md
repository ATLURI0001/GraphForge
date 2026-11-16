# Conditional Edge Routing & Cycle Prevention

## Routing Logic
Conditional edges evaluate the current agent state to determine the subsequent execution path:
- `should_continue()` checks whether required tools have resolved or if output schema is satisfied.
- `route_to_specialist()` directs queries to specialized domain subagents based on intent classification.

## Infinite Cycle Safeguards
Each thread maintains a `recursion_limit` (default: 25 steps). Exceeding this limit transitions the graph to an error recovery node.

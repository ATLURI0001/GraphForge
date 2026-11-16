# Sub-Agent Delegation & Error Recovery Patterns

## Topology
GraphForge employs a hierarchical Supervisor pattern:
1. **Supervisor Agent**: Deconstructs high-level queries into discrete domain subtasks.
2. **Worker Agents**: Research, Coding, and Verification agents execute concurrently.
3. **Aggregator Agent**: Synthesizes verified results and returns the canonical response.

## Error Recovery
Worker node failures trigger localized retries with exponential backoff before escalating to supervisor intervention.
